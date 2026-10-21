"""The tools a VLM drives the bimanual YAM with: it names objects and arms, the rest is planned here.

    seeing:  look, get_point_cloud, get_scene_state, get_grasps
    acting:  pick, place, move_to, set_gripper, hand_over, home

Built on capx_sparklab's YamSparkLabApi, which owns the ROS rig, cuRobo and
perception; this layer keeps the scene graph, makes grasps from each object's
bounding box (grasps.py), and does the bookkeeping a VLM forgets, such as attaching
what it holds. Registered with cap-x as ``TampApi``.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

import capx.integrations.franka.libero_sparklab_v2 as surface
import capx_sparklab.api as yam_api
from capx.integrations import register_api
from capx.integrations.base_api import ApiBase
from capx.integrations.vision.sam3 import init_sam3_point_prompt
from capx.utils.tool_envelope import err, ok
from capx_sparklab.api import YamCodeEnv, YamSparkLabApi
from capx_sparklab.robot import TOOL0_DEPTH

from . import grasps
from .scene import SceneGraph, box, largest_blob

ARMS = ("left", "right")
# Grasps on the table at most this far from a base: joint 2 holds about its rating there
# (DESIGN.md#cap-x-on-the-rig).
REACH = 0.35
LIFT = 0.10
# Heights above an object pick looks down from and approaches and lifts by, and above the
# release for place, tried highest first. Fingers down, an arm tops out near 0.35 m.
APPROACHES = (0.10, 0.05, 0.02)
# hand_over plays its plans this many times slower than real time; capx_sparklab's 2 lets the
# giver's wrist roll, kd/kp = 0.15 s and loaded, trail by YamEnv's 0.15 rad lag guard.
HANDOVER_SLOWDOWN = 3
# Released with its bottom this far above the surface, so it settles rather than strikes.
RELEASE_GAP = 0.02
# The real table top, 1.5 cm below the bases.
TABLE_Z = -0.015


def _matrix(pose: np.ndarray) -> np.ndarray:
    """4x4 from [x, y, z, qw, qx, qy, qz]."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.roll(pose[3:7], -1)).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def _pose(T: np.ndarray) -> dict[str, list[float]]:
    """A go_to_pose target from a 4x4."""
    return {"position": T[:3, 3].tolist(), "quaternion_xyzw": Rotation.from_matrix(T[:3, :3]).as_quat().tolist()}


class TampApi(ApiBase):
    """Task-level tools over one YamSparkLabApi.

    env: a capx_sparklab YamEnv.
    speak: let the robot API talk through the speaker.
    """

    def __init__(self, env: Any, speak: bool = False) -> None:
        super().__init__(env)
        self.robot = YamSparkLabApi(env, speak=speak)
        self._call = self.robot.functions()
        self.scene = SceneGraph(self.robot._clouds)
        self._point_prompt = init_sam3_point_prompt()

    def functions(self) -> dict[str, Any]:
        return {f.__name__: f for f in (self.look, self.get_point_cloud, self.get_scene_state, self.get_grasps,
                                        self.pick, self.place, self.move_to, self.set_gripper,
                                        self.hand_over, self.home)}

    # ---- seeing ------------------------------------------------------------------
    def look(self, cameras: list[str] | None = None) -> dict[str, Any]:
        """Capture the cameras now: "top" over the table, "left_wrist" and "right_wrist" on the grippers.

        Their images also arrive with every step; this says where each camera is and which way it faces.

        cameras: which ones, or None for all.
        Returns: {views: [{camera_id, image_id, resolution, pose, looks_along}]}.
        """
        return self._call["view_cameras"](camera_ids=cameras)

    def get_point_cloud(self, name: str, prompt: str | None = None, pixel: list[float] | None = None,
                        camera: str = "top") -> dict[str, Any]:
        """Find an object and remember it under a name; every other tool refers to it by that name.

        By prompt, every camera looks and the views are fused. By pointing, pixel is a
        point on the object in that camera's image as you see it, and that view alone is used.
        Call it again to refresh an object that moved; pick does so by itself.

        name: what to call it, e.g. "apple".
        prompt: how to pick it out ("the green apple left of the bowl"); defaults to name.
        pixel: [u, v] on the object in camera's image, instead of a prompt.
        camera: the camera pixel is on.
        Returns: {name, position, size, n_points}.
        """
        if pixel is not None:
            found = self._cloud_at_pixel(camera, pixel)
        else:
            found = self._call["perceive"](prompts=[prompt or name])
            if found["status"] == "ok":
                found = ok(**found["data"]["objects"][0]) if found["data"]["objects"][0]["found"] else err(
                    "NOT_FOUND", f"Could not find {prompt or name!r}: {found['data']['objects'][0].get('hint', '')}")
        if found["status"] != "ok":
            return found
        self.scene.add(name, prompt or name, found["data"]["cloud_id"])
        row = next(r for r in self.scene.describe() if r["name"] == name)
        return ok(name=name, position=row["position"], size=row["size"], n_points=len(self.scene.points(name)))

    def _cloud_at_pixel(self, camera: str, pixel: list[float]) -> dict[str, Any]:
        """SAM3's best mask around a pixel of a fresh frame, deprojected and stored."""
        frame = self.robot._resolve_source(camera)
        if frame is None:
            return err("NO_CAMERA", f"No frame from {camera!r}.")
        masks = self._point_prompt(frame["rgb"], [float(v) for v in pixel])
        if not masks:
            return err("NOT_FOUND", f"Nothing segmented at {pixel} in {camera}.")
        points = self.robot._masked_points(frame, masks[0]["mask"])
        if len(points) == 0:
            return err("NO_DEPTH", f"The object at {pixel} in {camera} has no depth.")
        cloud_id = f"cloud_point_{len(self.robot._clouds)}"
        self.robot._clouds[cloud_id] = {"points": points, "camera_id": camera}
        return ok(cloud_id=cloud_id)

    def get_scene_state(self) -> dict[str, Any]:
        """Everything known: each object's position, size and support, and each arm's gripper.

        Positions are in metres, in the frame at the left arm's base: x forward, y left, z up.
        The right arm's base is at y = -0.55. "on" names what an object rests on or in.

        Returns: {objects: [{name, position, size, top_z, held_by, on, seen_s_ago}],
        arms: {arm: {holding, gripper_opening_m, grasp_centre}}}.
        """
        arms = {}
        for arm in ARMS:
            tool = _matrix(self._env.tool_pose(arm))
            arms[arm] = {"holding": self.scene.held_by(arm),
                         "gripper_opening_m": round(self._env.joints(arm)[1], 3),
                         "grasp_centre": np.round(tool[:3, 3] + tool[:3, 2] * TOOL0_DEPTH, 3).tolist()}
        return ok(objects=self.scene.describe(), arms=arms)

    def get_grasps(self, name: str, arm: str | None = None) -> dict[str, Any]:
        """The top-down grasps pick would try on an object, and which of them the arm can carry out.

        name: a known object.
        arm: which arm; None picks the nearer one.
        Returns: {arm, plannable, proposed}.
        """
        if name not in self.scene.objects:
            return self._unknown(name)
        arm = arm or self._nearer_arm(name)
        candidates = self._grasp_candidates(name)
        if isinstance(candidates, dict):
            return candidates
        with self.robot._arm(arm):
            self.robot._refresh_world()
            ranked = self.robot._rank_pool(candidates)
        return ok(arm=arm, plannable=sum(c["plannable"] for c in ranked), proposed=len(ranked))

    def _grasp_candidates(self, name: str) -> list[dict] | dict[str, Any]:
        """The box grasps on name, registered with the robot API so go_to_pose can plan them."""
        found = grasps.top_down(self.scene.points(name), TABLE_Z)
        if isinstance(found, str):
            return err("TOO_WIDE", f"{name} is {found}.")
        cloud_id = self.scene.objects[name].cloud_id
        return [self.robot._store_grasp(g.position, g.rotation, g.score, cloud_id) for g in found]

    # ---- acting ------------------------------------------------------------------
    def pick(self, name: str, arm: str | None = None) -> dict[str, Any]:
        """Pick an object up and hold it up to 10 cm above where it was, less where the arm tops out.

        Finds it again, looks down at it through the wrist camera, offers the planner
        top-down grasps across its narrow side, closes, attaches it, and lifts. The
        fingers sense nothing: check the wrist camera image of this step shows the
        object between the fingers. If not, set_gripper(arm, 1.0) and pick again.

        name: a known object.
        arm: "left" or "right"; None picks the arm whose base is nearer.
        Returns: {arm, grasp_id, lifted, lift_note}: lift_note says why a lift stopped short.
        """
        if name not in self.scene.objects:
            return self._unknown(name)
        if self.scene.objects[name].held_by:
            return err("ALREADY_HELD", f"The {self.scene.objects[name].held_by} arm already holds {name}.")
        refreshed = self.get_point_cloud(name, self.scene.objects[name].prompt)
        if refreshed["status"] != "ok":
            return refreshed
        arm = arm or self._nearer_arm(name)
        if self.scene.held_by(arm):
            return err("HANDS_FULL", f"The {arm} arm holds {self.scene.held_by(arm)}; place it first.")
        distance = self._distance_from_base(name, arm)
        if distance > REACH:
            return err("OUT_OF_REACH", f"{name} is {distance:.2f} m from the {arm} arm's base; it reaches {REACH} m. "
                       "Try the other arm, or ask for it to be moved closer.")

        self._call["open_gripper"](arm_id=arm)
        self._look_from_above(name, arm)
        candidates = self._grasp_candidates(name)
        if isinstance(candidates, dict):
            return candidates
        grasp_ids = [c["grasp_id"] for c in candidates]
        cloud_id = self.scene.objects[name].cloud_id
        for clearance in APPROACHES:
            # Before the descent the open fingers straddle a tall object's top: it must not block them.
            self._call["update_collision_world"](ignore_cloud_ids=[cloud_id], arm_id=arm)
            # go_to_pose proves each grasp can be left by planning a LIFT_HEIGHT retreat too.
            with mock.patch.object(surface, "LIFT_HEIGHT", clearance):
                reached = self._call["go_to_pose"](arm_id=arm, grasp_ids=grasp_ids, z_approach=clearance)
            if reached["status"] == "ok":
                break
        else:
            return err("NO_GRASP", f"None of the {len(candidates)} grasps on {name} plans for the {arm} arm: "
                       f"{reached['message']} Try the other arm, or look again.")

        self._call["close_gripper"](arm_id=arm)
        self._call["update_collision_world"](attach_cloud_id=cloud_id, arm_id=arm)
        self.scene.objects[name].held_by = arm
        lifted = self._carry(name, arm, lambda: self._call["move_relative"](arm_id=arm, delta_xyz=[0, 0, clearance], frame="base"))
        return ok(arm=arm, grasp_id=reached["data"].get("reached_grasp_id"), lifted=lifted["status"] == "ok",
                  lift_note=None if lifted["status"] == "ok" else lifted["message"])

    def place(self, name: str, on: str | None = None, at: list[float] | None = None) -> dict[str, Any]:
        """Put a held object down, on or into another object, or at a point on the table.

        Carries it over the spot, lowers it until its bottom is 2 cm above the surface,
        opens, and backs off 10 cm up. The gripper keeps its current orientation. If
        the spot is beyond this arm, hand_over to the other arm first.

        name: the object held.
        on: an object to put it on or in; it is centred over that object's top.
        at: [x, y] on the table, or [x, y, z] over a surface at height z.
        Returns: {arm, released_over}.
        """
        if name not in self.scene.objects or not self.scene.objects[name].held_by:
            return err("NOT_HELD", f"Nothing holds {name!r}; pick it first.")
        if (on is None) == (at is None):
            return err("BAD_ARGUMENTS", "Give exactly one of on (an object) or at ([x, y] or [x, y, z]).")
        if on is not None and on not in self.scene.objects:
            return self._unknown(on)
        arm = self.scene.objects[name].held_by
        if on is not None:
            low, high = box(self.scene.points(on))
            spot, surface = (low[:2] + high[:2]) / 2, high[2]
        else:
            spot, surface = np.asarray(at[:2], dtype=float), float(at[2]) if len(at) == 3 else TABLE_Z

        # Where the gripper must be for the object's box to sit centred over spot, RELEASE_GAP clear.
        tool = _matrix(self._env.tool_pose(arm))
        low, high = box(self.scene.points(name))
        release = tool.copy()
        release[:2, 3] += spot - (low[:2] + high[:2]) / 2
        release[2, 3] += surface + RELEASE_GAP - low[2]

        # Come in from above the spot, as high as the arm reaches there.
        for clearance in APPROACHES:
            above = release.copy()
            above[2, 3] += clearance
            moved = self._carry(name, arm, lambda: self._call["go_to_pose"](arm_id=arm, pose=_pose(above)))
            if moved["status"] == "ok":
                break
        else:
            return err("NO_PLAN", f"The {arm} arm cannot bring {name} over {np.round(spot, 2).tolist()}: "
                       "out of its reach or blocked. hand_over it to the other arm, or place it nearer.")
        lowered = self._carry(name, arm, lambda: self._call["move_relative"](
            arm_id=arm, delta_xyz=[0, 0, -clearance], frame="base"))
        if lowered["status"] != "ok":
            return lowered
        self._release(name, arm)
        self._call["move_relative"](arm_id=arm, delta_xyz=[0, 0, LIFT], frame="base")
        return ok(arm=arm, released_over=np.round(spot, 3).tolist())

    def move_to(self, position: list[float], arm: str, yaw_deg: float = 0.0) -> dict[str, Any]:
        """Move an arm's grasp centre, the point between the fingertips, with the fingers pointing down.

        Whatever it holds comes along. The planner avoids what the cameras see.

        position: [x, y, z] in metres.
        arm: "left" or "right".
        yaw_deg: turn about the vertical; 0 has the fingers closing across y.
        Returns: {grasp_centre}.
        """
        T = np.eye(4)
        T[:3, :3] = (Rotation.from_euler("z", yaw_deg, degrees=True) * Rotation.from_euler("y", np.pi)).as_matrix()
        T[:3, 3] = np.asarray(position, dtype=float) - T[:3, 2] * TOOL0_DEPTH
        moved = self._carry(self.scene.held_by(arm), arm, lambda: self._call["go_to_pose"](arm_id=arm, pose=_pose(T)))
        if moved["status"] != "ok":
            return moved
        return ok(grasp_centre=np.round(np.asarray(position, dtype=float), 3).tolist())

    def set_gripper(self, arm: str, opening: float) -> dict[str, Any]:
        """Open (1) or close (0) a gripper, or anything between. Opening it lets go of what it holds.

        arm: "left" or "right".
        opening: 0 closed .. 1 fully open (9.4 cm).
        Returns: {opening_m, released}.
        """
        released = self.scene.held_by(arm) if opening >= 0.5 else None
        if released:
            self._release(released, arm)
        width = self._env.set_gripper(arm, float(np.clip(opening, 0.0, 1.0)))
        return ok(opening_m=round(width, 3), released=released)

    def hand_over(self, name: str) -> dict[str, Any]:
        """Pass a held object to the other arm, between the two, 0.3 m up.

        Nothing confirms the receiving grasp; check the wrist cameras afterwards.

        name: the object held.
        Returns: {holder}.
        """
        if name not in self.scene.objects or not self.scene.objects[name].held_by:
            return err("NOT_HELD", f"Nothing holds {name!r}; pick it first.")
        giver = self.scene.objects[name].held_by
        receiver = next(a for a in ARMS if a != giver)
        with mock.patch.object(yam_api, "SLOWDOWN", HANDOVER_SLOWDOWN):
            result = self._call["handover"](from_arm=giver, to_arm=receiver)
        if result["status"] == "ok":
            self.scene.objects[name].held_by = receiver
            return ok(holder=receiver)
        return result

    def home(self, arm: str) -> dict[str, Any]:
        """Move an arm to its home pose, out of the way. Whatever it holds comes along.

        arm: "left" or "right".
        Returns: {}.
        """
        return self._carry(self.scene.held_by(arm), arm, lambda: self._call["home"](arm_id=arm))

    # ---- bookkeeping ---------------------------------------------------------------
    def _carry(self, name: str | None, arm: str, motion) -> dict[str, Any]:
        """Run a motion of an arm; if it holds name, name's points move with the gripper."""
        before = _matrix(self._env.tool_pose(arm))
        result = motion()
        if name is not None:
            self.scene.move(name, _matrix(self._env.tool_pose(arm)) @ np.linalg.inv(before))
        return result

    def _release(self, name: str, arm: str) -> None:
        self._call["open_gripper"](arm_id=arm)
        self._call["update_collision_world"](detach_arm_id=arm)
        self.scene.objects[name].held_by = None

    def _look_from_above(self, name: str, arm: str) -> None:
        """Shift an object's cloud sideways onto the centre the wrist camera sees from straight above.

        The top camera sees an object's near side only, which pulls its centre up to
        2.5 cm toward the camera: more than the fingers' clearance on a 7 cm apple. Its
        cloud keeps the height, which the wrist camera, seeing mostly the top, cannot give.
        """
        low, high = box(self.scene.points(name))
        centre = (low + high) / 2
        for above in APPROACHES:
            if self.move_to([centre[0], centre[1], high[2] + above], arm)["status"] == "ok":
                break
        else:
            return
        frame = self.robot._resolve_source(f"{arm}_wrist")
        in_camera = np.linalg.inv(frame["pose_mat"]) @ [*centre, 1.0]
        pixel = (frame["intrinsics"] @ (in_camera[:3] / in_camera[2]))[:2]
        masks = self._point_prompt(frame["rgb"], pixel.tolist())
        points = self.robot._masked_points(frame, masks[0]["mask"]) if masks else np.empty((0, 3))
        if len(points) > 50:
            seen_low, seen_high = box(largest_blob(points))
            shift = np.eye(4)
            shift[:2, 3] = (seen_low[:2] + seen_high[:2]) / 2 - centre[:2]
            self.scene.move(name, shift)

    def _distance_from_base(self, name: str, arm: str) -> float:
        low, high = box(self.scene.points(name))
        return float(np.linalg.norm((low[:2] + high[:2]) / 2 - self._env.base_pose(arm)[:2, 3]))

    def _nearer_arm(self, name: str) -> str:
        return min(ARMS, key=lambda arm: self._distance_from_base(name, arm))

    def _unknown(self, name: str) -> dict[str, Any]:
        return err("UNKNOWN_OBJECT", f"No object {name!r}. Known: {list(self.scene.objects)}. Use get_point_cloud first.")


class TampCodeEnv(YamCodeEnv):
    """cap-x's code env on the YAM with the TAMP tools; the prompt comes from the config."""


register_api("TampApi", TampApi)
