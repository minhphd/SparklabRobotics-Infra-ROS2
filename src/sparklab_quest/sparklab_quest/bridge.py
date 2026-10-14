"""Quest bridge: serves the WebXR page and carries the headset onto ROS 2.

    ros2 run sparklab_quest bridge --ros-args -p certfile:=cert.pem -p keyfile:=key.pem \\
        -p rig:=src/sparklab_yam/config/cameras.yaml

    Quest page ── xr_frame ───► bridge ──► /quest/head/pose, /quest/<hand>/pose, /quest/<hand>/joy
               ── config ─────► bridge ──► /quest/settings  (latched JSON)
               ◄── ik_state ──  bridge ◄── /quest/<hand>/haptic, /quest/<hand>/engaged
               ◄══ WebRTC ════  bridge ◄── /cameras/<id>/color/image_raw

The page and the bridge speak JSON over one WebSocket (/ws), which also carries
WebRTC signalling; camera_toggle mutes a track by repeating its last frame.
Without a readable certfile/keyfile it serves plain HTTP, which the headset
accepts only on localhost (``adb reverse``).
"""

import asyncio
import json
import logging
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import av
import numpy as np
import rclpy
import uvicorn
import yaml
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from ament_index_python.packages import get_package_share_directory
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool, Float32, String

logger = logging.getLogger("quest_bridge")
WEB_DIR = Path(get_package_share_directory("sparklab_quest")) / "web"
HANDS = ("left", "right")
FEEDBACK_HZ = 30.0


class QuestBridge(Node):
    """The ROS side. WebSocket handlers call in from the asyncio thread."""

    def __init__(self) -> None:
        super().__init__("quest_bridge")
        self.host = self.declare_parameter("host", "0.0.0.0").value
        self.port = self.declare_parameter("port", 8443).value
        self.certfile = self.declare_parameter("certfile", "").value
        self.keyfile = self.declare_parameter("keyfile", "").value
        rig = self.declare_parameter("rig", "").value

        self._head_pub = self.create_publisher(PoseStamped, "/quest/head/pose", 10)
        self._pose_pubs = {h: self.create_publisher(PoseStamped, f"/quest/{h}/pose", 10)
                           for h in HANDS}
        self._joy_pubs = {h: self.create_publisher(Joy, f"/quest/{h}/joy", 10) for h in HANDS}
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._settings_pub = self.create_publisher(String, "/quest/settings", latched)
        self._settings: dict = {}

        self.haptic = dict.fromkeys(HANDS, 0.0)
        self.engaged = dict.fromkeys(HANDS, False)
        self.feedback_t = 0.0
        for h in HANDS:
            self.create_subscription(Float32, f"/quest/{h}/haptic",
                                     lambda m, h=h: self._on_feedback(self.haptic, h, m.data), 10)
            self.create_subscription(Bool, f"/quest/{h}/engaged",
                                     lambda m, h=h: self._on_feedback(self.engaged, h, m.data), 10)

        # id -> {label, shape}; frames[id] is the newest Image for that camera.
        self.cameras: dict[str, dict] = {}
        self.frames: dict[str, Image] = {}
        cams = (yaml.safe_load(Path(rig).read_text())["cameras"] or {}) if rig else {}
        for cam_id, cfg in cams.items():
            h = cfg.get("crop_height") or cfg.get("height", 480)
            self.cameras[cam_id] = {"label": cfg.get("label", cam_id.replace("_", " ").title()),
                                    "shape": (min(h, cfg.get("height", 480)), cfg.get("width", 640), 3)}
            self.create_subscription(Image, f"/cameras/{cam_id}/color/image_raw",
                                     lambda m, c=cam_id: self.frames.__setitem__(c, m), 1)

    def _on_feedback(self, table: dict, hand: str, value) -> None:
        table[hand] = value
        self.feedback_t = time.monotonic()

    def live_cameras(self) -> list[str]:
        """Configured cameras with a publisher, so a missing one disappears from the page."""
        return [c for c in self.cameras if self.count_publishers(f"/cameras/{c}/color/image_raw")]

    def publish_xr_frame(self, msg: dict) -> None:
        stamp = self.get_clock().now().to_msg()
        viewer = msg.get("viewer")
        if viewer:
            self._head_pub.publish(_pose_msg(stamp, viewer))
        for hand, ctrl in (msg.get("controllers") or {}).items():
            if hand not in HANDS or not ctrl:
                continue
            self._pose_pubs[hand].publish(_pose_msg(stamp, ctrl))
            joy = Joy()
            joy.header.stamp = stamp
            joy.header.frame_id = "quest"
            buttons = ctrl.get("buttons") or []
            joy.buttons = [int(bool(b.get("p"))) for b in buttons]
            joy.axes = [float(b.get("v", 0.0)) for b in buttons]
            self._joy_pubs[hand].publish(joy)

    def publish_settings(self, update: dict) -> None:
        """Merge a page update and republish the whole snapshot, latched."""
        self._settings.update(update)
        self._settings_pub.publish(String(data=json.dumps(self._settings)))


def _pose_msg(stamp, src: dict) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = "quest"
    p, q = src["position"], src["orientation"]  # WebXR orientation is xyzw
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
    (msg.pose.orientation.x, msg.pose.orientation.y,
     msg.pose.orientation.z, msg.pose.orientation.w) = map(float, q)
    return msg


class CameraTrack(VideoStreamTrack):
    """aiortc track over one camera topic. `enabled=False` repeats the last
    frame, so H.264 P-frames compress to nothing and no renegotiation is needed."""

    kind = "video"

    def __init__(self, bridge: QuestBridge, cam_id: str) -> None:
        super().__init__()
        self.bridge = bridge
        self.cam_id = cam_id
        self.enabled = True
        self._last = np.zeros(bridge.cameras[cam_id]["shape"], dtype=np.uint8)

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        msg = self.bridge.frames.get(self.cam_id)
        if self.enabled and msg is not None:
            self._last = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        vf = av.VideoFrame.from_ndarray(self._last, format="rgb24")
        vf.pts = pts
        vf.time_base = time_base
        return vf


def _prefer_h264(pc: RTCPeerConnection) -> None:
    # The Quest decodes H.264 in hardware; VP8 is software-decoded.
    h264 = [c for c in RTCRtpSender.getCapabilities("video").codecs if c.mimeType == "video/H264"]
    if h264:
        for transceiver in pc.getTransceivers():
            if transceiver.kind == "video":
                transceiver.setCodecPreferences(h264)


def build_app(bridge: QuestBridge) -> FastAPI:
    clients: dict[WebSocket, dict] = {}  # ws -> {"pc", "tracks"}

    async def send_feedback() -> None:
        """Push ik_state to every page while the teleop is publishing feedback."""
        while True:
            await asyncio.sleep(1.0 / FEEDBACK_HZ)
            if not clients or time.monotonic() - bridge.feedback_t > 1.0:
                continue
            text = json.dumps({"type": "ik_state",
                               **{f"{h}_haptic": float(bridge.haptic[h]) for h in HANDS},
                               **{f"{h}_engaged": bool(bridge.engaged[h]) for h in HANDS}})
            await asyncio.gather(*(ws.send_text(text) for ws in list(clients)),
                                 return_exceptions=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(send_feedback())
        yield
        task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def no_cache(request, call_next):
        # Operators should not have to clear the Quest's cache after a web change.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    async def close_pc(state: dict) -> None:
        if state.get("pc") is not None:
            await state["pc"].close()
        state["pc"], state["tracks"] = None, {}

    async def webrtc_request(ws: WebSocket, state: dict, msg: dict) -> None:
        await close_pc(state)
        live = bridge.live_cameras()
        enabled = set(msg.get("enabled_cameras") or live)
        pc = RTCPeerConnection()
        state["pc"] = pc
        for cam_id in live:
            track = CameraTrack(bridge, cam_id)
            track.enabled = cam_id in enabled
            # aiortc otherwise assigns a random stream id and the page cannot
            # match a track to its panel. Private attr, stable across aiortc 1.9+.
            pc.addTrack(track)._stream_id = cam_id
            state["tracks"][cam_id] = track
        _prefer_h264(pc)

        @pc.on("iceconnectionstatechange")
        async def on_ice() -> None:
            logger.info("ice state: %s", pc.iceConnectionState)
            if pc.iceConnectionState in ("failed", "closed"):
                await close_pc(state)

        await pc.setLocalDescription(await pc.createOffer())
        await ws.send_text(json.dumps({
            "type": "webrtc_offer", "sdp": pc.localDescription.sdp,
            "sdp_type": pc.localDescription.type,
            "cameras": [{"id": c, "label": bridge.cameras[c]["label"]} for c in live],
        }))

    @app.websocket("/ws")
    async def ws_handler(ws: WebSocket) -> None:
        await ws.accept()
        state = {"pc": None, "tracks": {}}
        clients[ws] = state
        peer = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
        logger.info("ws connect %s (now %d)", peer, len(clients))
        await ws.send_text(json.dumps({
            "type": "camera_list",
            "cameras": [{"id": c, "label": v["label"]} for c, v in bridge.cameras.items()],
        }))
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                t = msg.get("type")
                if t == "xr_frame":
                    bridge.publish_xr_frame(msg)
                elif t == "config_update":
                    bridge.publish_settings(msg.get("config") or {})
                elif t == "webrtc_request":
                    await webrtc_request(ws, state, msg)
                elif t == "webrtc_answer" and state["pc"] is not None:
                    await state["pc"].setRemoteDescription(
                        RTCSessionDescription(sdp=msg["sdp"], type=msg["sdp_type"]))
                elif t == "camera_toggle" and msg.get("camera_id") in state["tracks"]:
                    state["tracks"][msg["camera_id"]].enabled = bool(msg.get("enabled", True))
                elif t == "latency_report":
                    # From ?latency=1; `host` labels the transport (USB vs LAN).
                    logger.info("latency_report host=%s n=%s p50=%.1f p95=%.1f max=%.1f ms",
                                msg.get("host"), msg.get("n"), msg.get("p50", 0.0),
                                msg.get("p95", 0.0), msg.get("max", 0.0))
                elif t == "ping":
                    await ws.send_json({"echo": msg, "server_time": time.time()})
                # ice_candidate is dropped: aiortc puts every local candidate in the offer.
        except WebSocketDisconnect:
            logger.info("ws disconnect %s", peer)
        finally:
            await close_pc(state)
            clients.pop(ws, None)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def _lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # no packet is sent; this picks the route
        return s.getsockname()[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # uvicorn owns SIGINT/SIGTERM; ROS is shut down after it returns.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    bridge = QuestBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    stop = threading.Event()

    def spin() -> None:
        while not stop.is_set():
            executor.spin_once(timeout_sec=0.1)

    # Joined before executor.shutdown(), which aborts the process if a spin
    # is still waiting.
    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()

    tls = bool(bridge.certfile and bridge.keyfile
               and Path(bridge.certfile).is_file() and Path(bridge.keyfile).is_file())
    if not tls:
        logger.warning("no readable certfile/keyfile — serving HTTP; the headset needs HTTPS "
                       "except over adb reverse (make one with scripts/make_certs.sh)")
    logger.info("Quest browser -> %s://%s:%d/", "https" if tls else "http", _lan_ip(), bridge.port)
    try:
        uvicorn.run(build_app(bridge), host=bridge.host, port=bridge.port, log_level="warning",
                    ssl_certfile=bridge.certfile if tls else None,
                    ssl_keyfile=bridge.keyfile if tls else None)
    finally:
        stop.set()
        spinner.join()
        executor.shutdown()
        bridge.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
