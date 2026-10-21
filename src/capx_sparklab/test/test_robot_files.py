"""The generated robot description is consistent with the gripper constants; no GPU needed."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from capx_sparklab.robot import CUROBO_CONFIG, GRASPGENX_GRIPPER, GRASPGENX_ROOT, TOOL0_DEPTH, TOOL_FRAME, URDF


def test_urdf_has_the_planning_frames():
    links = {link.get("name") for link in ET.parse(URDF).getroot().findall("link")}
    assert {"world", TOOL_FRAME, "tool0", "tip_left", "tip_right"} <= links
    tool0 = ET.parse(URDF).getroot().find("joint[@name='tool0_joint']/origin").get("xyz").split()
    assert float(tool0[2]) == TOOL0_DEPTH


def test_curobo_config_covers_every_link():
    kinematics = yaml.safe_load(CUROBO_CONFIG.read_text())["kinematics"]
    assert kinematics["tool_frames"] == [TOOL_FRAME]
    meshed = {"base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"}
    assert meshed <= set(kinematics["collision_spheres"])
    assert kinematics["extra_collision_spheres"] == {"attached_object": 4}


def test_graspgenx_gripper_is_the_stock_one():
    config = json.loads((GRASPGENX_ROOT / GRASPGENX_GRIPPER / "config.json").read_text())
    assert abs(config["fingertip"][2] - TOOL0_DEPTH) < 1e-3
    assert config["sweep_volume"]["offset"][2] == config["fingertip"][2]


def test_graspgenx_collision_mesh_is_the_open_hand():
    config = json.loads((GRASPGENX_ROOT / GRASPGENX_GRIPPER / "config.json").read_text())
    lines = (GRASPGENX_ROOT / GRASPGENX_GRIPPER / "coll_mesh.obj").read_text().splitlines()
    vertices = np.array([[float(v) for v in line.split()[1:4]] for line in lines if line.startswith("v ")])
    # GraspGenX's frame: the fingers open along x and point along z.
    assert np.ptp(vertices[:, 0]) > config["sweep_volume"]["extents"][0]
    assert vertices[:, 2].max() > config["fingertip"][2]
