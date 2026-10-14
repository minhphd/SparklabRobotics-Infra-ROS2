"""WebXR "xr-standard" button indices, as carried on ``/quest/<hand>/joy``.

``joy.buttons[i]`` is button i's pressed flag and ``joy.axes[i]`` its analog
value. A controller reporting fewer buttons reads as unpressed past its end.
"""

from __future__ import annotations

TRIGGER = 0      # analog — gripper closure
GRIP = 1         # clutch
THUMBSTICK = 3   # press — stow this arm
A_X = 4          # precision mode
B_Y = 5          # arm/disarm, or start/end an episode when recording

# Past this, a controller pose predates the gap and acting on it would
# produce catch-up motion.
XR_FRAME_STALE_TIMEOUT_S = 0.2


def pressed(buttons, index: int) -> bool:
    return index < len(buttons) and bool(buttons[index])


def value(axes, index: int) -> float:
    return float(axes[index]) if index < len(axes) else 0.0
