"""Set the held object straight down below the gripper (keeping its orientation), release,
rise clear, go to rest.   usage: place_down.py ARM TABLE_Z [H_PLACE] [H_CLEAR]"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rig-toolkit" / "scripts"))
from skill_core import *

arm, tz = sys.argv[1], float(sys.argv[2])
h_place = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
h_clear = float(sys.argv[4]) if len(sys.argv) > 4 else 0.085
p, qq = get_ee_pose(arm)
R = quat_to_R(qq)
t = p + FINGERTIPS * R[:, 2]
stamp(f"start: tips {np.round(t, 3)} ({t[2]-tz:.3f} above table), grip {get_joint_angles(arm)[6]:.3f}")
fast_line(arm, [t[0], t[1], tz + h_place], R, tol=0.004, label="lower")
st = command(arm, np.r_[get_joint_angles(arm)[:6], 1.0])
stamp(f"released, grip {st[6]:.3f}")
fast_line(arm, [t[0], t[1], tz + h_clear], R, tol=0.008, label="rise")
fast_move(arm, REST, min_tip_z=tz + 0.06, label="rest")
Image.fromarray(get_image("top")).save("place_down_top.png")
stamp("done")
