"""Per-stage timing of a real rollout tick.

Answers one question: of the ~60 ms a control tick costs, how much is the
POLICY and how much is everything else? Every previous number for inference in
this repo was derived by subtraction; this measures it.

    PYTHONPATH=src python scripts/profile_rollout.py \\
        --policy.path=../SparkRobot/MolmoAct2/lerobot_training/last/pretrained_model \\
        --robot.type=yam_ultra_sim --robot.port=8081 --ticks 90

Takes the same flags as ``lerobot-rollout`` (they go to LeRobot's own parser
untouched), plus ``--ticks`` and ``--warmup``.

HOW IT MEASURES

The real components are wrapped, not reimplemented: ``preprocessor``,
``policy.select_action``, ``postprocessor``, ``robot.get_observation`` and
``robot.send_action`` each get a timing shim. So what is timed is exactly what
the rollout runs -- a reimplementation would measure a loop nobody uses.

The tick is driven as fast as it will go, with no ``precise_sleep``: the
question is what the loop COSTS, not whether it hits a target it is currently
being paced to.

WHY INFERENCE TICKS ARE SEPARATED

``select_action`` pops from a queue and only runs the model when that queue is
empty -- once every ``n_action_steps`` policy actions. Averaging over all ticks
hides a bimodal distribution: ~1 s on one tick, ~0 ms on the next 29. Both
numbers matter, and they answer different questions (peak stall vs achievable
mean rate), so both are reported.

CUDA IS ASYNCHRONOUS. Without a synchronize, the timer stops when the kernels
are QUEUED, not when they finish, and inference looks impossibly fast while
the cost reappears in whatever runs next. Every stage boundary syncs.
"""

from __future__ import annotations

import statistics as st
import sys
import time


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class TimedProxy:
    """Callable stand-in that times ``__call__`` and forwards everything else.

    Needed for the pre/post processors: they are called as objects but ALSO
    carry methods the engine uses (``reset()``), so replacing them with a bare
    function breaks ``engine.reset()``. Delegating via ``__getattr__`` keeps
    the object's full surface intact while still timing the call.
    """

    def __init__(self, inner, timer, label):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_timer", timer)
        object.__setattr__(self, "_label", label)

    def __call__(self, *a, **kw):
        _sync()
        t0 = time.perf_counter()
        try:
            return self._inner(*a, **kw)
        finally:
            _sync()
            t = self._timer
            t.cur[self._label] = t.cur.get(self._label, 0.0) + (time.perf_counter() - t0) * 1000

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


class Timer:
    """Accumulates per-stage durations, tagged by tick."""

    def __init__(self):
        self.rows: list[dict] = []
        self.cur: dict = {}

    def instrument_pipeline(self, pipeline, prefix):
        """Time each step inside a processor pipeline individually.

        Uses the pipeline's own ``before_step_hooks`` / ``after_step_hooks``
        rather than wrapping the steps, so nothing about the pipeline's
        structure changes -- ``save_pretrained`` and the step list stay intact.

        Labels are ``<prefix>[NN].<StepClassName>``. The index is part of the
        label, not decoration: a pipeline may hold two instances of the same
        step class, and keying on the class name alone silently sums them into
        one row that looks like a single expensive step. Zero-padding keeps the
        report's ``sorted()`` in pipeline order rather than alphabetical.

        These are reported separately from the ``preprocess`` / ``postprocess``
        totals, not summed into the tick, so the existing columns keep meaning
        what they meant.
        """
        steps = getattr(pipeline, "steps", None)
        if not steps:
            print(f"  ! {prefix}: no .steps to instrument")
            return
        names = [f"{prefix}[{i:02d}].{s.__class__.__name__}"
                 for i, s in enumerate(steps)]
        state = {"t0": 0.0}

        def before(idx, _transition):
            _sync()
            state["t0"] = time.perf_counter()

        def after(idx, _transition):
            _sync()
            dt = (time.perf_counter() - state["t0"]) * 1000
            self.cur[names[idx]] = self.cur.get(names[idx], 0.0) + dt

        pipeline.register_before_step_hook(before)
        pipeline.register_after_step_hook(after)
        return names

    def proxy(self, obj, attr, label):
        """Wrap ``obj.attr`` in a :class:`TimedProxy`, preserving its API."""
        setattr(obj, attr, TimedProxy(getattr(obj, attr), self, label))

    def wrap(self, obj, attr, label):
        """Replace ``obj.attr`` with a timing shim recording into ``label``.

        Replaces the ATTRIBUTE, never ``__call__`` on the target. Python looks
        up dunders on the type, not the instance, so shimming ``__call__`` on a
        preprocessor object is silently ignored and the stage reads 0.00 ms --
        which looks like a fast pipeline rather than a broken probe.
        """
        fn = getattr(obj, attr)

        def timed(*a, **kw):
            _sync()
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                _sync()
                self.cur[label] = self.cur.get(label, 0.0) + (time.perf_counter() - t0) * 1000

        try:
            setattr(obj, attr, timed)
        except Exception as e:                       # frozen dataclass, slots…
            print(f"  ! cannot instrument {attr}: {type(e).__name__}: {e}")
            return fn
        return fn


def _extract(argv):
    own = {"--ticks": ("ticks", int), "--warmup": ("warmup", int)}
    out, rest, i = {}, [], 0
    while i < len(argv):
        key, _, inline = argv[i].partition("=")
        if key in own:
            name, cast = own[key]
            if inline:
                out[name] = cast(inline); i += 1
            else:
                out[name] = cast(argv[i + 1]); i += 2
            continue
        rest.append(argv[i]); i += 1
    return out, rest


def main() -> int:
    own, rest = _extract(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]
    ticks = own.get("ticks", 90)
    warmup = own.get("warmup", 5)

    import threading

    from lerobot.configs import parser
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.rollout.context import build_rollout_context
    from lerobot.utils.action_interpolator import ActionInterpolator
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.utils import init_logging

    register_third_party_plugins()

    T = Timer()

    def _teardown(ctx):
        """Stop the engine and disconnect the robot — mirrors LeRobot's own
        ``RolloutStrategy._teardown_hardware``.

        Not optional, and not only about tidiness. Without it:

        * the arms are left holding whatever pose the last action commanded,
          instead of parked at zero. They do not drop (the drivers own the
          torque and outlive this process), but that is not the documented safe
          state and the next run starts from somewhere arbitrary.
        * ``cam.disconnect()`` never runs, so librealsense's C++ capture threads
          are still live when the interpreter tears down, a destructor fires on
          a joinable thread, and the process ends in
          ``terminate called without an active exception`` / SIGABRT. That core
          dump also hides any real error the run was about to report.
        """
        try:
            engine = ctx.policy.inference
            if engine is not None:
                engine.stop()
        except Exception:
            print("  ! engine.stop() failed")
        try:
            robot = ctx.hardware.robot_wrapper.inner
            if robot.is_connected:
                # YamUltraFollower.disconnect() parks both arms to zero first,
                # then releases the cameras.
                robot.disconnect()
        except Exception as e:
            print(f"  ! robot.disconnect() failed: {type(e).__name__}: {e}")
            print("  ! the arms may be left holding their last commanded pose")

    def _run(cfg):
        init_logging()
        cfg.display_data = False          # visualisation is not what we profile
        ctx = build_rollout_context(cfg, threading.Event())

        engine = ctx.policy.inference
        robot = ctx.hardware.robot_wrapper
        interp = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
        features = ctx.data.dataset_features
        ordered_keys = ctx.data.ordered_action_keys

        policy = engine._policy
        T.wrap(policy, "select_action", "policy")
        # The pre/post processors are called as objects, so the shim goes on
        # the ENGINE's reference to them, not on the objects themselves.
        T.proxy(engine, "_preprocessor", "preprocess")
        T.proxy(engine, "_postprocessor", "postprocess")
        T.wrap(robot, "get_observation", "get_obs")
        T.wrap(robot, "send_action", "send_act")
        T.proxy(ctx.processors, "robot_observation_processor", "obs_proc")
        T.proxy(ctx.processors, "robot_action_processor", "act_proc")
        # Per-step breakdown inside the policy pipelines. Sub-step labels carry
        # a "." so the tick accounting below can tell them from the top-level
        # stages and not double-count them into `other`.
        T.instrument_pipeline(engine._preprocessor, "pre")
        T.instrument_pipeline(engine._postprocessor, "post")

        engine.reset()
        engine.start()
        engine.resume()

        print(f"\nfps target {cfg.fps}   interpolation x{cfg.interpolation_multiplier}"
              f"   n_action_steps {getattr(cfg.policy, 'n_action_steps', '?')}"
              f"   chunk {getattr(cfg.policy, 'chunk_size', '?')}")
        print(f"profiling {ticks} ticks ({warmup} warmup, discarded)\n")

        hdr = ("tick", "total", "get_obs", "obs_proc", "preproc", "policy",
               "postproc", "act_proc", "send_act", "other", "INF")
        print("".join(f"{h:>9}" for h in hdr))

        cached = None
        try:
            _profile_loop(cfg, ctx, engine, robot, interp, policy, features,
                          ordered_keys, ticks, warmup)
        finally:
            _teardown(ctx)

    def _profile_loop(cfg, ctx, engine, robot, interp, policy, features,
                      ordered_keys, ticks, warmup):
        cached = None
        for i in range(ticks + warmup):
            T.cur = {}
            _sync()
            t0 = time.perf_counter()

            obs = robot.get_observation()
            if cached is None or interp.needs_new_action():
                cached = ctx.processors.robot_observation_processor(obs)

            was_inference = False
            if interp.needs_new_action():
                frame = build_dataset_frame(features, cached, prefix=OBS_STR)
                # Queue empty => this call runs the model, not just a pop.
                q = getattr(policy, "_action_queue", None)
                was_inference = not q
                act = engine.get_action(frame)
                if act is not None:
                    interp.add(act.cpu())

            out = interp.get()
            if out is not None:
                ad = {k: out[j].item() for j, k in enumerate(ordered_keys)}
                robot.send_action(ctx.processors.robot_action_processor((ad, obs)))

            _sync()
            total = (time.perf_counter() - t0) * 1000
            if i < warmup:
                continue

            row = dict(T.cur)
            row["total"] = total
            row["inference"] = was_inference
            # Only the top-level stages account for the tick; sub-step labels
            # (which contain ".") are a breakdown OF preprocess/postprocess and
            # would double-count into `other` if summed here.
            acc = sum(v for k, v in T.cur.items() if "." not in k)
            row["other"] = total - acc
            T.rows.append(row)

            def g(k):
                return row.get(k, 0.0)

            print(f"{i - warmup:>9}{total:9.1f}{g('get_obs'):9.1f}"
                  f"{g('obs_proc'):9.1f}{g('preprocess'):9.1f}{g('policy'):9.1f}"
                  f"{g('postprocess'):9.1f}{g('act_proc'):9.1f}{g('send_act'):9.1f}"
                  f"{row['other']:9.1f}{'  <==' if was_inference else '':>9}")

        _report(T.rows, cfg)

    _run.__annotations__["cfg"] = RolloutConfig
    parser.wrap()(_run)()
    return 0


def _report(rows, cfg):
    if not rows:
        print("no rows")
        return
    inf = [r for r in rows if r["inference"]]
    non = [r for r in rows if not r["inference"]]
    stages = ["get_obs", "obs_proc", "preprocess", "policy", "postprocess",
              "act_proc", "send_act", "other"]

    print("\n" + "=" * 78)
    print(f"{len(rows)} ticks: {len(inf)} ran the model, {len(non)} reused the chunk")

    def block(name, rs):
        if not rs:
            return
        print(f"\n{name} ({len(rs)} ticks)")
        print(f"  {'stage':<14}{'mean ms':>10}{'p50':>9}{'max':>9}{'% of tick':>11}")
        tot = st.mean([r["total"] for r in rs])
        for s in stages:
            v = [r.get(s, 0.0) for r in rs]
            print(f"  {s:<14}{st.mean(v):10.2f}{sorted(v)[len(v)//2]:9.2f}"
                  f"{max(v):9.2f}{100*st.mean(v)/tot:10.1f}%")
        print(f"  {'TOTAL':<14}{tot:10.2f}{sorted([r['total'] for r in rs])[len(rs)//2]:9.2f}"
              f"{max(r['total'] for r in rs):9.2f}")

    block("INFERENCE ticks", inf)
    block("chunk-reuse ticks", non)

    def substeps(name, rs):
        """Per-step breakdown inside the policy pipelines.

        Reported against the pipeline's own total rather than the tick, because
        the question this answers is "which step inside the 12 ms" -- and shown
        for chunk-reuse ticks specifically, since that is where the work is
        discarded and so where removing it is free.
        """
        if not rs:
            return
        keys = sorted({k for r in rs for k in r if "." in k})
        if not keys:
            return
        print(f"\n{name} — per-step ({len(rs)} ticks)")
        print(f"  {'step':<44}{'mean ms':>10}{'p50':>9}{'max':>9}")
        for k in keys:
            v = [r.get(k, 0.0) for r in rs]
            print(f"  {k:<44}{st.mean(v):10.2f}{sorted(v)[len(v)//2]:9.2f}{max(v):9.2f}")
        tot = sum(st.mean([r.get(k, 0.0) for r in rs]) for k in keys)
        print(f"  {'(sum of steps)':<44}{tot:10.2f}")

    substeps("chunk-reuse ticks", non)
    substeps("INFERENCE ticks", inf)

    mean_all = st.mean([r["total"] for r in rows])
    hz = 1000.0 / mean_all
    print("\n" + "-" * 78)
    print(f"mean tick {mean_all:.1f} ms  ->  {hz:.1f} Hz sustained")

    pol = st.mean([r.get("policy", 0.0) for r in rows])
    rest = mean_all - pol
    print(f"  policy (amortised)  {pol:7.1f} ms   {100*pol/mean_all:.0f}%")
    print(f"  everything else     {rest:7.1f} ms   {100*rest/mean_all:.0f}%")
    if inf:
        print(f"  one inference costs {st.mean([r.get('policy', 0.0) for r in inf]):.0f} ms, "
              f"amortised over {len(rows)/max(len(inf),1):.0f} ticks")

    print("\nwhat 30 Hz (33.3 ms) requires:")
    print(f"  need to remove {mean_all - 33.3:.1f} ms per tick")
    if pol > 33.3:
        print(f"  NOTE: amortised policy alone is {pol:.1f} ms — above the whole")
        print(f"        budget. No amount of sim tuning reaches 30 Hz; the model")
        print(f"        must get ~{pol/ (33.3 - rest) if 33.3 > rest else float('inf'):.1f}x faster"
              if 33.3 > rest else
              "        must get faster AND the rest must shrink.")
    else:
        print(f"  policy fits ({pol:.1f} ms); trim {rest - (33.3 - pol):.1f} ms from the rest")


if __name__ == "__main__":
    raise SystemExit(main())
