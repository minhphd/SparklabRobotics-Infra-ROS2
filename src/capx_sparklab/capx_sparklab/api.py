"""cap-x's SparkLab tool surface for the two YAM arms, and the code env that serves it.

The 13 tools of cap-x's SparkLabApi, each addressed to one arm, plus
``handover``. Both arms plan in one world frame (left_base): each has its own
cuRobo planner, built from capx_sparklab.robot with its base placed from the
rig's TF, and its own fused world, in which the other arm is just more depth.
Perception, clouds and grasps are shared.

Registered with cap-x as ``YamSparkLabApi``; importing this module registers it.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import capx.integrations.franka.libero_sparklab_v2 as surface
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.integrations import register_api
from capx.integrations.motion.curobo_v2 import Planner, WorldMapper
from capx.integrations.vision import graspgenx
from capx.utils.geometry import quat_wxyz, quat_xyzw
from capx.utils.tool_envelope import err, ok, pose_in, pose_out

from .env import MAX_LAG, YamEnv
from .robot import GRASPGENX_GRIPPER, GRASPGENX_ROOT, curobo_config

# Two planners share the GPU with perception; cap-x reads this as its top_k ceiling too.
surface.PLAN_BATCH = 32

TABLE = {"name": "table", "dims": [3.0, 3.0, 0.1], "pose": [0.3, -0.3, -0.06, 1.0, 0.0, 0.0, 0.0]}
"""Top 1 cm below the bases; the calibration board read 1.3-1.8 cm below, so this errs high."""

FUSION_SIZE = (640, 480)
"""Every frame is resampled to this before fusion: cuRobo's mapper and robot segmenter
take one resolution, and the top camera's (1280x720) is not the wrists'."""

REACH = 0.35
"""m across the table from an arm's base to an object it grasps: joint 2 holds about its 9 N m
rating there. See DESIGN.md#cap-x-on-the-rig."""

SLOWDOWN = 2
"""Plans play this many times slower than cuRobo times them, on the same path.
See DESIGN.md#cap-x-on-the-rig."""

PER_ARM = ("_planner", "_mapper", "_arm_spheres", "_home_joints", "_payload")
"""SparkLabApi state that belongs to one arm, swapped in while a tool runs on it."""
ARM_TOOLS = (*surface.MOTION_TOOLS, "get_grasp_poses", "open_gripper", "close_gripper", "lift")
REPLANNING = (*surface.MOTION_TOOLS, "get_grasp_poses")
"""Tools that plan, so first refresh the arm's world: the other arm may have moved since."""

HANDOVER_HEIGHT = 0.30
HANDOVER_FORWARD = 0.30
HANDOVER_CLEARANCE = 0.06
"""The receiving grasp centre stays this far from the giver's, clear of its fingers."""
HANDOVER_RETREAT = 0.05
"""The giver backs out this far; with its fingers open, 8 cm folds it past what plans."""
HANDOVER_ROLLS = np.radians([0, 90, -90])
HANDOVER_ANGLE = np.radians(60)
"""The giver's level fingers turn this far from forward toward the receiver; with
HANDOVER_FORWARD, the one pose that suits both arms. See DESIGN.md#cap-x-on-the-rig."""


class _Planner(Planner):
    """cap-x's Planner, keeping the lift of the grasp go_to_pose executes: its first reachable.

    That lift is planned with the fingers exempt, as the descent is, so it can leave a grasp
    low on the table, where the fingertips sit inside the table's margin and nothing else plans.
    """

    lift: np.ndarray | None = None

    def plan_grasps(self, poses: list, start_q: np.ndarray, approach: float, lift: float,
                    disable_collision_links: list[str] | None = None) -> Any:
        batch = super().plan_grasps(poses, start_q, approach, lift, disable_collision_links)
        chosen = next((i for i, reached in enumerate(batch.reachable) if reached), None)
        # Plain pose plans come through here too, with no lift; lift checks it starts where the arm is.
        if lift and chosen is not None:
            self.lift = batch.lift[chosen]
        return batch


class _Printed:
    """Stands in for cap-x's spoken Voice when it cannot run: lines go to stdout."""

    def say(self, text: str) -> None:
        print(f"[robot] {text}", flush=True)

    def wait(self, timeout: float = 30.0) -> None:
        pass


class YamSparkLabApi(surface.SparkLabApi):
    """SparkLabApi on both YAM arms.

    env: a YamEnv.
    speak: say verbalize lines aloud through cap-x's Voice (a TTS model and an audio
        device); they are printed instead when False, or when it cannot start.
    Remaining keyword arguments go to SparkLabApi.
    """

    def __init__(self, env: YamEnv, speak: bool = True, **kwargs: Any) -> None:
        spoken = surface.Voice

        def voice() -> Any:
            if not speak:
                return _Printed()
            try:
                return spoken()
            except Exception as exc:
                print(f"[robot] cannot speak ({type(exc).__name__}: {exc}); printing lines instead")
                return _Printed()

        # GraspGenX finds a gripper, and its collision mesh, under GRIPPER_ASSETS at call time.
        graspgenx.GRIPPER_ASSETS = GRASPGENX_ROOT
        with mock.patch.object(surface, "Voice", voice), mock.patch.object(
                surface, "init_graspgenx", lambda: graspgenx.init_graspgenx(GRASPGENX_GRIPPER)):
            super().__init__(env, arms=env.arms, **kwargs)
        self._view = env.scene
        self._per_arm = {
            arm: {"_planner": None, "_mapper": None, "_arm_spheres": 0,
                  "_home_joints": env.joints(arm)[0], "_payload": self._payload}
            for arm in self._arms
        }
        self._active = self._arms[0]
        self._load(self._active)

    # ---- one arm at a time ---------------------------------------------------
    @contextlib.contextmanager
    def _arm(self, arm: str) -> Iterator[None]:
        """Run the enclosed calls on one arm's planner, world, home and payload."""
        previous = self._active
        self._stash(previous)
        self._load(arm)
        try:
            yield
        finally:
            self._stash(arm)
            self._load(previous)

    def _stash(self, arm: str) -> None:
        self._per_arm[arm] = {key: getattr(self, key) for key in PER_ARM}

    def _load(self, arm: str) -> None:
        for key, value in self._per_arm[arm].items():
            setattr(self, key, value)
        self._active = arm
        self._env.active_arm = arm

    def _enveloped(self, fn: Callable[..., Any], name: str = "") -> Callable[..., Any]:
        """SparkLabApi's envelope, run on the arm the call names.

        The arm is activated outside the envelope, so the world it restores
        after a motion is that arm's.
        """
        if name not in ARM_TOOLS:
            return super()._enveloped(fn, name)
        signature, plain = inspect.signature(fn), fn

        @functools.wraps(plain)
        def refreshed(*args: Any, **kwargs: Any) -> Any:
            if name in REPLANNING and self._planner is not None:
                self._refresh_world()
            return plain(*args, **kwargs)

        enveloped = super()._enveloped(refreshed, name)

        @functools.wraps(plain)
        def on_arm(*args: Any, **kwargs: Any) -> Any:
            try:
                arm = signature.bind_partial(*args, **kwargs).arguments.get("arm_id")
            except TypeError:
                arm = None
            if arm not in self._arms:
                return enveloped(*args, **kwargs)  # which reports the missing or unknown arm
            with self._arm(arm):
                return enveloped(*args, **kwargs)

        return on_arm

    def _tools(self) -> dict[str, Callable[..., Any]]:
        return {**super()._tools(), "handover": self.handover, "lift": self.lift}

    def _check_arm(self, arm_id: str | None) -> str | dict[str, Any]:
        if arm_id is None:
            return err("ARM_REQUIRED", f"Say which arm: one of {list(self._arms)}.")
        return super()._check_arm(arm_id)

    # ---- the robot under SparkLabApi -------------------------------------------
    @property
    def planner(self) -> Planner:
        """The active arm's planner, built on first use: seconds of CUDA graph capture."""
        if self._planner is None:
            self._planner = _Planner(robot=curobo_config(self._env.base_pose(self._active)),
                                    batch=surface.PLAN_BATCH)
            self._planner.warmup()
            self._mapper = WorldMapper(self._planner)
            self._arm_spheres = len(self._planner.collision_spheres(self._joint_positions()))
            self._refresh_world()
        return self._planner

    def _joint_positions(self) -> np.ndarray:
        """The active arm's joints, inside the planner's limits.

        At rest joints 2 and 3 read a hair below their 0 stops, and cuRobo refuses
        to plan from a start outside its limits ("Start or End state in collision").
        """
        q = super()._joint_positions()
        if self._planner is None:
            return q
        return np.clip(q, *self._planner.joint_limits())

    def _execute(self, waypoints: np.ndarray) -> None:
        waypoints = np.asarray(waypoints, dtype=float)
        steps = np.arange(len(waypoints))
        slowed = np.linspace(0, len(waypoints) - 1, (len(waypoints) - 1) * SLOWDOWN + 1)
        self._env.execute_trajectory(self._active, np.column_stack([np.interp(slowed, steps, j) for j in waypoints.T]))
        self._refresh_world()

    def _set_gripper(self, fraction: float) -> float:
        return self._env.set_gripper(self._active, fraction)

    def _refresh_world(self) -> None:
        """Fuse current depth into the active arm's world, the table and what it holds.

        Only this arm's own pixels are masked out of the depth, so the other arm
        enters this arm's world as an obstacle. Its own wrist camera is left out.
        """
        planner = self._planner or self.planner
        # Masked, the fingers' ray-shadow fills this arm's own wrist view, and fusion
        # surfaces it against the fingers; see DESIGN.md#cap-x-on-the-rig.
        frames = [_resampled(f) for f in self._live_frames() if f["camera_id"] != f"{self._active}_wrist"]
        carve = list(self._ignored.values())
        held = self._attached.get(self._active)
        if held in self._clouds:
            # The map keeps what it saw, so the held object's own voxels would enclose it.
            points = self._clouds[held]["points"]
            carve.append((points.min(axis=0) - surface.IGNORE_PAD, points.max(axis=0) + surface.IGNORE_PAD))
        field = self._mapper.update(frames, self._joint_positions(), ignore=carve) if frames else None
        planner.set_world([dict(TABLE)], voxels=field)
        held = self._attached.get(self._active)
        if held is None:
            planner.detach()
        else:
            dims, pose = self._payload
            planner.attach_box(held, dims, pose, self._joint_positions())

    def _moving_cameras(self, obs: dict[str, Any]) -> set[str]:
        tools = [self._env.tool_pose(arm)[:3] for arm in self._arms]
        return {c for c in self._camera_ids(obs)
                if min(np.linalg.norm(np.asarray(obs[c]["pose"])[:3] - t) for t in tools)
                < surface.WRIST_MOUNT_RADIUS}

    # ---- tools that differ from SparkLabApi --------------------------------------
    # cap-x's own copy binds quat_wxyz to an array, then calls it: every move that succeeds
    # raises after moving. The same code with that local renamed; drop once cap-x is fixed.
    def move_relative(
        self,
        arm_id: str | None = None,
        delta_xyz: list[float] | None = None,
        delta_rpy: list[float] | None = None,
        frame: str = "ee",
        force_threshold: float | None = None,
        speed: str = "slow",
        max_distance: float | None = None,
    ) -> dict[str, Any]:
        """Straight-line motion composed from the current end-effector pose.

        "down 3 cm" or "retreat along the approach".
        ``frame="ee"`` moves along the gripper's own axes, ``"base"`` along world
        axes. With ``force_threshold`` set it advances until contact or
        ``max_distance``: stopping on contact is a result, not an error, and
        ``stopped_on_contact`` says which happened.

        Every probe is planned, so it will not drive into anything the cameras
        have mapped — a guarded descent onto the table returns an error rather
        than contact. Contact only fires for what the depth map missed, which
        makes this a probe for the unmapped, not a substitute for planning.

        Hard rule: prefer this over hand-computed absolute poses for small
        adjustments — that is where sign and axis errors live.

        arm_id: which arm.
        delta_xyz: [dx, dy, dz] in metres.
        delta_rpy: [roll, pitch, yaw] in radians.
        frame: "ee" or "base".
        force_threshold: stop when contact exceeds this; None travels the full delta.
        Returns: {final_pose, travel, stopped_on_contact, contact_force, restored}.
        """
        arm = self._check_arm(arm_id)
        if isinstance(arm, dict):
            return arm
        if frame not in ("ee", "base"):
            return err("BAD_ARGUMENTS", f"frame must be 'ee' or 'base', not {frame!r}.")

        delta = np.asarray(delta_xyz or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        rpy = np.asarray(delta_rpy or [0.0, 0.0, 0.0], dtype=float).reshape(3)

        cartesian = np.asarray(self._env.get_observation()["robot_cartesian_pos"], dtype=float)
        start = cartesian[:3].copy()
        rotation = Rotation.from_quat(quat_xyzw(cartesian))

        world_delta = rotation.apply(delta) if frame == "ee" else delta
        target_rotation = (
            rotation * Rotation.from_euler("xyz", rpy)
            if frame == "ee"
            else Rotation.from_euler("xyz", rpy) * rotation
        )
        target_quat = target_rotation.as_quat()
        target_wxyz = np.array(
            [target_quat[3], target_quat[0], target_quat[1], target_quat[2]], dtype=float
        )

        if force_threshold is None:
            steps = [start + world_delta]
        else:
            steps = self._contact_steps(start, world_delta, max_distance)

        stopped_on_contact = False
        for step_target in steps:
            before = np.asarray(
                self._env.get_observation()["robot_cartesian_pos"], dtype=float
            )[:3]
            try:
                path = self.planner.plan_to_pose(
                    step_target, target_wxyz, self._joint_positions()
                )
                if path is None:
                    raise RuntimeError(
                        "no collision-free path to the next step of this motion"
                    )
                self._execute(path)
            except Exception as exc:
                return err("IK_INFEASIBLE", f"Cannot reach {step_target.tolist()}: {exc}")
            after = np.asarray(
                self._env.get_observation()["robot_cartesian_pos"], dtype=float
            )[:3]
            # The arm not closing the gap it was commanded to close is our only
            # contact signal; the simulator exposes no wrist force.
            if force_threshold is not None and np.linalg.norm(after - before) < 1e-4:
                stopped_on_contact = True
                break

        restored = self._restore_ignored()
        end = np.asarray(self._env.get_observation()["robot_cartesian_pos"], dtype=float)
        return ok(
            final_pose=pose_out(end[:3], quat_wxyz(end)),
            travel=float(np.linalg.norm(end[:3] - start)),
            stopped_on_contact=stopped_on_contact,
            contact_force=None,
            restored=restored,
        )

    def get_state(self, arm_id: str | None = None) -> dict[str, Any]:
        """Current joint positions, end-effector pose, gripper and attachments.

        joint_limits are the lower and upper bound of each joint, in the same
        order as joint_positions, and move_to_joints refuses anything outside
        them. ee_pose is the gripper base; the fingertips' grasp centre is 0.20 m
        further along its z axis.

        arm_id: "left" or "right", or None for both.
        Returns: {arms: {id: {joint_positions, joint_names, joint_limits,
        ee_pose, gripper_width, attached_cloud_id}}, cameras}.
        """
        if arm_id is not None and arm_id not in self._arms:
            return err("UNKNOWN_ARM", f"No arm {arm_id!r}. Available: {list(self._arms)}.")
        arms = {}
        for arm in self._arms if arm_id is None else (arm_id,):
            with self._arm(arm):
                arms[arm] = super().get_state(arm)["data"]["arms"][arm]
        return ok(arms=arms, cameras=self._camera_ids(self._env.get_observation()))

    def close_gripper(self, arm_id: str | None = None, force: float | None = None) -> dict[str, Any]:
        """Close the gripper and report the width it actually reached.

        The fingers sense nothing, so the width alone does not prove anything
        is held: look. view_cameras straight after closing —
        this arm's wrist camera should show the object between the fingers.
        Then attach it with update_collision_world (nothing this arm tries plans
        until you do), lift it out with lift, and look again: it should have
        come up off the table. If it has not, detach it, open, and re-perceive.

        arm_id: which arm.
        force: reserved; the gripper closes at a fixed effort.
        Returns: {final_width, effort}.
        """
        arm = self._check_arm(arm_id)
        if isinstance(arm, dict):
            return arm
        return ok(final_width=self._set_gripper(0.0), effort=None)

    def get_grasp_poses(self, cloud_id: str, arm_id: str | None = None, top_k: int = 20) -> dict[str, Any]:
        arm = self._check_arm(arm_id)
        cloud = self._clouds.get(cloud_id)
        if cloud is not None and not isinstance(arm, dict):
            centre = (cloud["points"].min(axis=0) + cloud["points"].max(axis=0)) / 2
            distance = float(np.linalg.norm(centre[:2] - self._env.base_pose(arm)[:2, 3]))
            if distance > REACH:
                return err("OUT_OF_REACH", f"{cloud_id} is {distance:.2f} m from the {arm} arm's base; past {REACH} m "
                           "its shoulder strains holding it. Use the other arm, or ask for it to be moved closer.")
        return super().get_grasp_poses(cloud_id, arm_id, top_k)

    get_grasp_poses.__doc__ = surface.SparkLabApi.get_grasp_poses.__doc__

    def lift(self, arm_id: str | None = None) -> dict[str, Any]:
        """Back out of the grasp just made, 10 cm along the approach, as go_to_pose planned it.

        The way out of a grasp: low on the table, the fingertips sit inside the table's
        margin and every other motion fails to plan. Close and attach first. Works only
        straight after go_to_pose with grasp_ids; any other motion in between forgets it.

        arm_id: which arm.
        Returns: {final_pose}.
        """
        arm = self._check_arm(arm_id)
        if isinstance(arm, dict):
            return arm
        path = self.planner.lift
        # Closing deflects the wrist a little; past the lag guard's bound, it is another pose's lift.
        if path is None or not len(path) or np.max(np.abs(path[0] - self._joint_positions())) > MAX_LAG:
            return err("NO_LIFT", "No grasp to back out of: lift works only straight after "
                       "go_to_pose with grasp_ids, before any other motion.")
        self._execute(path)
        self.planner.lift = None
        return ok(final_pose=self.get_state(arm)["data"]["arms"][arm]["ee_pose"])

    def update_collision_world(
        self,
        attach_cloud_id: str | None = None,
        detach_arm_id: str | None = None,
        ignore_cloud_ids: list[str] | None = None,
        restore_cloud_ids: list[str] | None = None,
        arm_id: str | None = None,
    ) -> dict[str, Any]:
        """Hand an object to one arm, take one back, or edit the world for one motion.

        ``attach_cloud_id`` with ``arm_id`` records that this arm holds the
        object. Do it as soon as a grasp is verified: until then the object is an
        obstacle with that gripper inside it, and nothing that arm tries plans.
        ``detach_arm_id`` releases what that arm holds, once it is placed.

        ``ignore_cloud_ids`` deletes objects from the world for exactly one
        motion of either arm, so it may touch them on purpose — push, nudge,
        press; the next go_to_pose, move_relative, move_to_joints or home puts
        them back and lists them in ``restored``. What is deleted is the object's
        bounding box, so a close neighbour goes with it, and the arm drives
        through that space at full speed.

        Each arm avoids the other as the cameras see it, so keep both in view.
        Known limit: the carried object is not yet collision-checked.

        attach_cloud_id: cloud of the object now held, with arm_id.
        detach_arm_id: arm to release.
        ignore_cloud_ids: clouds to delete from the world for the next motion.
        restore_cloud_ids: clouds to put back now, without moving.
        arm_id: the arm attach_cloud_id goes to.
        Returns: {obstacles, attachments, ignored}.
        """
        for cloud_id in [attach_cloud_id, *(ignore_cloud_ids or []), *(restore_cloud_ids or [])]:
            if cloud_id is not None and cloud_id not in self._clouds:
                return err("UNKNOWN_CLOUD", f"No cloud {cloud_id!r}. Known: {list(self._clouds)}.")
        for arm in (detach_arm_id, arm_id):
            if arm is not None and arm not in self._arms:
                return err("UNKNOWN_ARM", f"No arm {arm!r}. Available: {list(self._arms)}.")
        if attach_cloud_id is not None and arm_id is None:
            return err("ARM_REQUIRED", f"Say which arm holds {attach_cloud_id}: arm_id, one of {list(self._arms)}.")

        for cloud_id in restore_cloud_ids or []:
            self._ignored.pop(cloud_id, None)
        for cloud_id in ignore_cloud_ids or []:
            points = self._clouds[cloud_id]["points"]
            self._ignored[cloud_id] = (points.min(axis=0) - surface.IGNORE_PAD,
                                       points.max(axis=0) + surface.IGNORE_PAD)
        if detach_arm_id is not None:
            self._attached.pop(detach_arm_id, None)
            with self._arm(detach_arm_id):
                self._refresh_world()
        if attach_cloud_id is not None:
            with self._arm(arm_id):
                self._attached[arm_id] = attach_cloud_id
                self._payload = self._payload_in_tool_frame(attach_cloud_id)
                self._refresh_world()
        return ok(obstacles=["depth_field", TABLE["name"]], attachments=dict(self._attached),
                  ignored=list(self._ignored))

    def handover(self, from_arm: str, to_arm: str, height: float = HANDOVER_HEIGHT) -> dict[str, Any]:
        """Pass the object one gripper holds to the other gripper.

        The giving arm must hold an attached object (update_collision_world with
        attach_cloud_id and arm_id after a verified grasp). It brings the object
        between the arms with its fingers level and pointing at the receiving
        arm, so the object sticks out toward it; the receiving arm opens and
        grasps it at least 6 cm from the giver's fingers; the giver lets go and
        backs off. The receiver ends holding it, attached. Nothing moves if the
        meeting point cannot be reached. Nothing checks the receiving grasp
        either: view_cameras afterwards to see which gripper has the object.
        Works best for an object held near one end.

        from_arm: the arm holding the object.
        to_arm: the arm to receive it.
        height: metres above the bases to pass it at.
        Returns: {cloud_id, holder, grasp_id, final_width, retreated}.
        """
        if {from_arm, to_arm} - set(self._arms) or from_arm == to_arm:
            return err("BAD_ARGUMENTS", f"from_arm and to_arm must be different arms of {list(self._arms)}.")
        cloud_id = self._attached.get(from_arm)
        if cloud_id is None:
            return err("NOT_HOLDING", f"The {from_arm} arm holds nothing attached. Grasp the object, "
                       "then update_collision_world(attach_cloud_id=..., arm_id=...) first.")
        if self._attached.get(to_arm):
            return err("HANDS_FULL", f"The {to_arm} arm already holds {self._attached[to_arm]}. Place it first.")

        with self._arm(from_arm):
            self._refresh_world()
            before = self._tool_matrix()
            meet = self._meeting_poses(from_arm, to_arm, height)
            batch = self.planner.plan_grasps(meet, self._joint_positions(), approach=0.0, lift=0.0)
            chosen = next((i for i, reach in enumerate(batch.reachable) if reach), None)
            if chosen is None:
                return err("NO_PLAN", f"The {from_arm} arm cannot bring the object between the arms near "
                           f"{np.round(meet[0][0], 2).tolist()}. Try another height, or home it first.")
            self._execute(batch.approach[chosen])
            moved = self._tool_matrix() @ np.linalg.inv(before)
            cloud = self._clouds[cloud_id]
            cloud["points"] = cloud["points"] @ moved[:3, :3].T + moved[:3, 3]
            giver = self._grasp_centre(self._tool_matrix())

        with self._arm(to_arm):
            self._refresh_world()
            self._set_gripper(1.0)
            proposed = self._graspgenx_candidates(cloud["points"], surface.PLAN_BATCH, cloud_id)
            candidates = [c for c in proposed
                          if np.linalg.norm(self._grasp_centre(self._pose_matrix(c["pose"])) - giver)
                          >= HANDOVER_CLEARANCE]
            reachable = [c for c in self._rank_pool(candidates) if c["plannable"]]
            if not reachable:
                return err("NO_PLAN", f"No grasp on {cloud_id} plans for the {to_arm} arm: {len(proposed)} "
                           f"proposed, {len(candidates)} clear of the {from_arm} gripper, none reachable. "
                           "Try another height.")
            batch = self.planner.plan_grasps([pose_in(c["pose"]) for c in reachable], self._joint_positions(),
                                             approach=surface.PREGRASP_STANDOFF, lift=surface.LIFT_HEIGHT)
            chosen = next((i for i, reach in enumerate(batch.reachable) if reach), None)
            if chosen is None:
                return err("NO_PLAN", f"The {to_arm} arm's grasps stopped planning; try again.")
            self._execute(batch.approach[chosen])
            self._execute(batch.grasp[chosen])
            width = self._set_gripper(0.0)

        with self._arm(from_arm):
            self._set_gripper(1.0)
            self._attached.pop(from_arm)
            self._refresh_world()
        with self._arm(to_arm):
            self._attached[to_arm] = cloud_id
            self._payload = self._payload_in_tool_frame(cloud_id)
            self._refresh_world()
        with self._arm(from_arm):
            self._refresh_world()
            retreat = self.move_relative(arm_id=from_arm, delta_xyz=[0.0, 0.0, -HANDOVER_RETREAT], frame="ee")
        return ok(cloud_id=cloud_id, holder=to_arm, grasp_id=reachable[chosen]["grasp_id"],
                  final_width=width, retreated=retreat["status"] == "ok")

    # ---- handover geometry -----------------------------------------------------
    def _meeting_poses(self, giver: str, receiver: str, height: float) -> list[tuple[np.ndarray, np.ndarray]]:
        """The giver's grasp_base poses for passing, preferred first.

        The grasp centre sits HANDOVER_FORWARD in front of the bases' midpoint,
        at height and then 5 cm either side, with the fingers level and turned
        HANDOVER_ANGLE toward the receiver, so the object sticks out on its side.
        """
        start, end = self._env.base_pose(giver)[:3, 3], self._env.base_pose(receiver)[:3, 3]
        side = end - start
        side[2] = 0.0
        side /= np.linalg.norm(side)
        forward = np.cross(side, [0.0, 0.0, 1.0]) * (1 if np.cross(side, [0.0, 0.0, 1.0])[0] > 0 else -1)
        toward = np.cos(HANDOVER_ANGLE) * forward + np.sin(HANDOVER_ANGLE) * side
        down = np.array([0.0, 0.0, -1.0])
        level = Rotation.from_matrix(np.column_stack([down, np.cross(toward, down), toward]))
        poses = []
        for h in (height, height - 0.05, height + 0.05):
            centre = (start + end) / 2 + [HANDOVER_FORWARD, 0.0, h]
            for roll in HANDOVER_ROLLS:
                x, y, z, w = (level * Rotation.from_euler("z", roll)).as_quat()
                poses.append((centre - toward * self._reach, np.array([w, x, y, z])))
        return poses[:surface.PLAN_BATCH]

    def _tool_matrix(self) -> np.ndarray:
        cartesian = self._env.tool_pose(self._active)
        return self._pose_matrix({"position": cartesian[:3], "quaternion_xyzw": [*cartesian[4:], cartesian[3]]})

    @staticmethod
    def _pose_matrix(pose: dict[str, Any]) -> np.ndarray:
        position, wxyz = pose_in(pose)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([*wxyz[1:], wxyz[0]]).as_matrix()
        T[:3, 3] = position
        return T

    def _grasp_centre(self, tool: np.ndarray) -> np.ndarray:
        return tool[:3, 3] + tool[:3, 2] * self._reach


def _resampled(frame: dict[str, Any], size: tuple[int, int] = FUSION_SIZE) -> dict[str, Any]:
    """A camera frame at another resolution, intrinsics scaled to match; depth by nearest pixel."""
    height, width = np.asarray(frame["depth"]).shape[:2]
    if (width, height) == size:
        return frame
    sx, sy = size[0] / width, size[1] / height
    K = np.array(frame["intrinsics"], dtype=float)
    K[0, 0], K[0, 2] = K[0, 0] * sx, (K[0, 2] + 0.5) * sx - 0.5
    K[1, 1], K[1, 2] = K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5
    return {**frame, "intrinsics": K,
            "depth": cv2.resize(np.asarray(frame["depth"]), size, interpolation=cv2.INTER_NEAREST),
            "rgb": cv2.resize(np.asarray(frame["rgb"]), size, interpolation=cv2.INTER_AREA)}


class YamCodeEnv(CodeExecutionEnvBase):
    """cap-x's code env on the YAM rig; the prompt comes from the config."""


# The fingers read 0 m apart when they meet on nothing.
register_api("YamSparkLabApi", YamSparkLabApi)
