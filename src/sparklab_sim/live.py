"""Edit a scene ``.usda`` in your editor, orbit it in a browser, over SSH.

    ./scripts/isaac_python.sh -m sparklab_sim.live

Boots Isaac once and serves an interactive viewport on one TCP port. Drag to
orbit, wheel to zoom, shift-drag to pan. Save the ``.usda`` in any editor and
the view updates in place — ``.usda`` is ASCII, so moving a prim is editing
its ``xformOp:translate``. The camera dropdown switches between free look and
the cameras authored in the scene, so you can jump from "what am I moving" to
"what does the rig actually see".

The scene file is loaded as a sublayer of an anonymous wrapper stage rather
than opened directly::

    wrapper stage (anonymous root)
      |- session layer : free camera, overview camera   (never saved)
      `- sublayer      : scene.usda                     (the file you edit)

Reloading is then ``Sdf.Layer.Reload()`` on that sublayer, which leaves the
Replicator render products alone — ~0.04 s per edit against ~0.38 s for a full
reopen, and against the segfault that reloading the root layer causes. See
DESIGN.md. ``--reopen`` forces the slow path.

``--play`` makes the viewer re-render continuously; it does not start physics,
which is always running. Leave it off while placing things.
"""

from __future__ import annotations

import argparse
import hashlib
import signal
import sys
import time
from pathlib import Path

# Consecutive steady polls before acting, so a non-atomic editor write cannot
# reload a half-written file. Costs one poll of latency.
STABLE_POLLS = 2

FREE = "__free__"          # sentinel the browser sends for the free-look camera


class LayerWatcher:
    """Fingerprints every on-disk layer the stage composes from.

    Watching only the scene file would miss edits to the assets it references,
    which change what renders without touching its mtime. The watch list comes
    from ``Stage.GetUsedLayers()`` and is rebuilt on every reload.

    The fingerprint is (mtime_ns, size), since some filesystems quantise mtime
    to a second. ``--hash`` switches to content hashing for network mounts.
    """

    def __init__(self, use_hash: bool = False):
        self.use_hash = use_hash
        self.paths: list[Path] = []
        self._last: dict[Path, object] = {}
        self._stable_count = 0
        self._pending: dict[Path, object] | None = None

    def track(self, paths: list[Path]) -> None:
        self.paths = paths
        self._last = self._snapshot()
        self._pending = None
        self._stable_count = 0

    def _fingerprint(self, p: Path):
        try:
            st = p.stat()
        except OSError:
            return None
        if not self.use_hash:
            return (st.st_mtime_ns, st.st_size)
        try:
            return hashlib.blake2b(p.read_bytes(), digest_size=16).hexdigest()
        except OSError:
            return None

    def _snapshot(self) -> dict[Path, object]:
        return {p: self._fingerprint(p) for p in self.paths}

    def changed(self) -> list[Path]:
        """Layers whose contents have settled since the last change.

        Returns paths rather than a bool, so the caller can reload precisely
        those layers instead of re-reading every payload tree.
        """
        now = self._snapshot()
        if now == self._last:
            self._pending = None
            self._stable_count = 0
            return []

        if now == self._pending:
            self._stable_count += 1
            if self._stable_count >= STABLE_POLLS:
                hit = [p for p, f in now.items() if self._last.get(p) != f]
                self._last = now
                self._pending = None
                self._stable_count = 0
                return hit
        else:
            self._pending = now
            self._stable_count = 1
        return []


def _disk_layers(stage) -> list[Path]:
    """Resolved on-disk paths for every layer in the stage's composition."""
    out = []
    for layer in stage.GetUsedLayers():
        ident = layer.realPath or layer.identifier
        if not ident or ident.startswith(("omniverse://", "anon:")):
            continue
        p = Path(ident)
        if p.exists():
            out.append(p)
    return sorted(set(out))


def _dispose(renderers: dict) -> None:
    """Release every render product. See ``render.Renderer.close``."""
    for r in renderers.values():
        try:
            r.close()
        except Exception as e:
            print(f"  dispose: {type(e).__name__}: {e}")


def main() -> int:
    # Left running under tmux with output redirected, where Python would
    # block-buffer stdout and swallow every reload message.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None,
                    help="scene .usda (default: sparklab_yam/sim/scene.usda)")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to reach it from another machine directly")
    ap.add_argument("--poll", type=float, default=0.1,
                    help="seconds between change checks. Save-to-pixels is "
                         "STABLE_POLLS * poll + reload; with reload down at "
                         "~0.12 s the debounce is now the larger half, which "
                         "is why this is 0.1 and not 0.5")
    ap.add_argument("--settle", type=int, default=6,
                    help="app updates per frame when idle (quality)")
    ap.add_argument("--drag-settle", type=int, default=1,
                    help="app updates per frame while orbiting (latency)")
    ap.add_argument("--res", default="960x540",
                    help="free-look render resolution, WxH")
    ap.add_argument("--cam-res", default="640x360",
                    help="resolution for cameras authored in the scene. "
                         "Defaults to the dataset's 640x360 so a rendered "
                         "frame is directly comparable to a recorded one; "
                         "rendering a rig camera at the free-look resolution "
                         "would change its aspect and make that comparison a lie")
    ap.add_argument("--play", action="store_true",
                    help="re-render continuously (default: render only on "
                         "change). Does NOT start physics — the timeline is "
                         "always playing and gravity is off; see the module "
                         "docstring")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="render rate with --play")
    ap.add_argument("--hash", action="store_true",
                    help="content-hash instead of mtime; for network filesystems")
    ap.add_argument("--reopen", action="store_true",
                    help="full open_stage per change instead of sublayer "
                         "reload; slower but recomposes from scratch")
    ap.add_argument("--once", action="store_true",
                    help="render one frame, write it, exit (for CI / snapshots)")
    ap.add_argument("--out", default="scene_preview.png",
                    help="output path for --once")
    ap.add_argument("--regenerate", action="store_true",
                    help="rebuild the scene file from rig.py, DISCARDING edits")
    args = ap.parse_args()

    try:
        rw, rh = (int(v) for v in args.res.lower().split("x"))
    except ValueError:
        ap.error(f"--res wants WxH, got {args.res!r}")
    try:
        cw, ch = (int(v) for v in args.cam_res.lower().split("x"))
    except ValueError:
        ap.error(f"--cam-res wants WxH, got {args.cam_res!r}")

    # --- Isaac boots here; no isaacsim/omni import may precede this ---------
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Sdf, Usd

    from . import paths, render, scene, viewport

    scene_path = Path(args.scene) if args.scene else (
        paths.YAM_ULTRA_MODEL_DIR.parent / "sim" / "scene.usda")

    if args.regenerate or not scene_path.exists():
        why = "regenerating" if scene_path.exists() else "no scene file yet"
        print(f"{why} -> building {scene_path} from rig.py")
        scene.export_usd(scene_path)

    watcher = LayerWatcher(use_hash=args.hash)
    renderers: dict[str, render.Renderer] = {}
    state = {"cam": None, "active": FREE}

    def authored(st) -> list[str]:
        """Cameras that came from the file, for the browser's dropdown.

        Excludes the free-look camera, which is ours and already has an entry.
        """
        return list(authored_paths(st))

    def authored_paths(st) -> dict:
        return {n: p for n, p in scene.cameras(st).items()
                if p != scene.FREE_PRIM}

    # ---- stage construction ------------------------------------------------
    def open_wrapper() -> None:
        """Anonymous root stage with the scene file as its only sublayer.

        The indirection is the point: the scene file can then be reloaded
        independently of the stage carrying the render products.
        """
        stage_utils.create_new_stage()
        st = stage_utils.get_current_stage()
        st.GetRootLayer().subLayerPaths = [str(scene_path)]
        app.update()

    def author_cameras() -> None:
        """Add the free-look camera to the session layer.

        Session-layer edits are never saved, so this cannot leak into the file
        being edited and survives a sublayer reload untouched.
        """
        st = stage_utils.get_current_stage()
        with Usd.EditContext(st, st.GetSessionLayer()):
            # Re-apply the browser's last pose so a reload does not throw the
            # user back to the default view mid-edit.
            scene.set_orbit_camera(st, scene.FREE_PRIM, **(state["cam"] or {}))

    def build_renderer(name: str) -> None:
        """(Re)build the single render product for the active camera.

        One camera, not all: a fresh Kit stage carries four stock viewport
        cameras, and rendering them all costs five frames per update.
        """
        nonlocal renderers
        _dispose(renderers)
        renderers = {}
        st = stage_utils.get_current_stage()
        path = (scene.FREE_PRIM if name == FREE
                else scene.cameras(st).get(name, scene.FREE_PRIM))
        if not st.GetPrimAtPath(path).IsValid():
            name, path = FREE, scene.FREE_PRIM
        res = (rw, rh) if name == FREE else (cw, ch)
        renderers = {name: render.Renderer(app, path, resolution=res,
                                           settle_frames=args.settle)}

    def load(full: bool = True) -> bool:
        """Open (or reopen) the wrapper stage and rebuild the renderer.

        Guarded end to end: a scene that parses but fails during start() would
        otherwise take the process down mid-edit.
        """
        try:
            if full:
                _dispose(renderers)
                open_wrapper()
            scene.start(app)
            author_cameras()
            build_renderer(state["active"])
            st = stage_utils.get_current_stage()
            watcher.track(_disk_layers(st))
            found = authored(st)
            print(f"  cameras: {', '.join(found) or '(none in file)'}"
                  f"   watching {len(watcher.paths)} layer(s)")
            return True
        except Exception as e:
            print(f"  load failed: {type(e).__name__}: {e}")
            return False

    def reload_layers(changed: list[Path]) -> bool:
        """Fast path: reload just the layers that changed on disk.

        Falls back to a full reopen when a changed file is not a layer of this
        composition — a newly added reference, say.
        """
        if args.reopen:
            return load(full=True)
        try:
            hit = 0
            for p in changed:
                layer = Sdf.Layer.Find(str(p))
                if layer is not None:
                    layer.Reload(force=True)
                    hit += 1
            if hit != len(changed):
                return load(full=True)      # something new appeared
            app.update()
            # PhysX owns dynamic poses and overwrites the authored value, so
            # without a bounce an edited prop position does nothing on screen.
            scene.start(app, restart=True)
            author_cameras()                # re-assert in case the file moved it
            st = stage_utils.get_current_stage()
            watcher.track(_disk_layers(st))
            return True
        except Exception as e:
            print(f"  sublayer reload failed ({type(e).__name__}: {e}); reopening")
            return load(full=True)

    def frame(settle: int):
        for r in renderers.values():
            try:
                return r.frame(settle=settle)
            except Exception as e:
                print(f"  render failed: {type(e).__name__}: {e}")
        return None

    # --- one-shot mode: no server, no loop ---------------------------------
    if args.once:
        try:
            if not load():
                return 1
            img = frame(args.settle)
            if img is None:
                print("nothing rendered")
                return 1
            try:
                import imageio.v3 as iio
                iio.imwrite(args.out, img)
            except ImportError:
                from PIL import Image
                Image.fromarray(img).save(args.out)
            print(f"wrote {args.out}")
            return 0
        finally:
            _dispose(renderers)
            app.close()

    vp = viewport.Viewport(port=args.port, host=args.host,
                           home=scene.HOME_VIEW)
    url = vp.start()
    print(f"\n  scene : {scene_path}")
    print(f"  view  : {url}")
    print(f"  mode  : {'continuous render' if args.play else 'static (render on change)'}"
          f"   reload: {'full reopen' if args.reopen else 'sublayer'}")
    print("  drag to orbit, wheel to zoom, shift-drag to pan.")
    print("  edit the .usda and save — the view follows. Ctrl-C to stop.\n")

    # SIGTERM too: run under tmux and killed by scripts, where skipping the
    # finally block leaks the port and the GPU context.
    stopping = False

    def _stop(signum, frame_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)

    try:
        if not load():
            print("initial load failed; fix the scene and save to retry")
        else:
            img = frame(args.settle)
            if img is not None:
                vp.publish(img)

        st = stage_utils.get_current_stage()
        vp.set_info(authored(st), "live")

        next_render = time.monotonic()
        next_poll = 0.0
        last_move = 0.0

        while not stopping:
            now = time.monotonic()
            drew = False

            # --- file changes ------------------------------------------------
            if now >= next_poll:
                next_poll = now + args.poll
                changed = watcher.changed()
                if vp.take_reload():
                    changed = changed or [scene_path]
                if changed or not renderers:
                    names = ", ".join(p.name for p in changed) or "(forced)"
                    print(f"[{time.strftime('%H:%M:%S')}] changed: {names}")
                    t = time.time()
                    ok = reload_layers(changed) if renderers else load(full=True)
                    if ok:
                        img = frame(args.settle)
                        if img is not None:
                            vp.publish(img)
                            drew = True
                        st = stage_utils.get_current_stage()
                        vp.set_info(authored(st),
                                    f"reloaded in {time.time()-t:.2f}s")

            # --- camera switch from the browser ------------------------------
            sel = vp.take_selection()
            if sel is not None and sel != state["active"]:
                state["active"] = sel
                build_renderer(sel)
                img = frame(args.settle)
                if img is not None:
                    vp.publish(img)
                    drew = True

            # --- orbit input --------------------------------------------------
            cam = vp.take_camera()
            if cam is not None:
                state["cam"] = cam
                if state["active"] == FREE:
                    st = stage_utils.get_current_stage()
                    with Usd.EditContext(st, st.GetSessionLayer()):
                        scene.set_orbit_camera(st, scene.FREE_PRIM, **cam)
                    img = frame(args.drag_settle)
                    if img is not None:
                        vp.publish(img)
                        drew = True
                    last_move = now

            # --- quality refinement once the mouse stops ----------------------
            # Orbiting renders at drag_settle, which RTX sampling makes noisy.
            if (not args.play and last_move and now - last_move > 0.35
                    and state["active"] == FREE):
                last_move = 0.0
                img = frame(args.settle)
                if img is not None:
                    vp.publish(img)
                    drew = True

            # --- physics ------------------------------------------------------
            if args.play:
                if now >= next_render:
                    next_render = now + 1.0 / max(args.fps, 0.1)
                    img = frame(args.drag_settle)
                    if img is not None:
                        vp.publish(img)
                        drew = True
            elif not drew:
                # Nothing changes on its own, and the MJPEG stream holds the last
                # frame. Short sleep: this is also the orbit input latency.
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        _dispose(renderers)
        vp.stop()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
