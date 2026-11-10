"""Arm and disarm both GELLO nodes of teleop.launch.py with one call.

    ros2 service call /gello/arm std_srvs/srv/Trigger       # both arms, or neither
    ros2 service call /gello/disarm std_srvs/srv/Trigger

Forwards to /gello/left/{arm,disarm} and /gello/right/{arm,disarm}. If either arm
refuses, both are disarmed again, so one arm never follows its leader alone.
"""

from __future__ import annotations

import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from sparklab_yam.constants import HANDS
from sparklab_yam.run import run

CALL_TIMEOUT_S = 3.0
"""s to wait on a GELLO node; its own arm request waits up to 1 s for a leader reading."""


class GelloPair(Node):
    def __init__(self) -> None:
        super().__init__("gello")
        group = ReentrantCallbackGroup()
        self._triggers = {(h, op): self.create_client(Trigger, f"/gello/{h}/{op}", callback_group=group)
                          for h in HANDS for op in ("arm", "disarm")}
        self.create_service(Trigger, "~/arm", self._on_arm, callback_group=group)
        self.create_service(Trigger, "~/disarm", self._on_disarm, callback_group=group)
        self.get_logger().info("both arms follow their leaders after `ros2 service call /gello/arm "
                               "std_srvs/srv/Trigger`; /gello/disarm stops both")

    def _call(self, op: str) -> dict[str, Trigger.Response]:
        """Call op on both GELLO nodes at once.

        Returns: each arm's reply; a node that is not up or does not answer reads as a refusal.
        """
        futures = {h: self._triggers[h, op].call_async(Trigger.Request())
                   for h in HANDS if self._triggers[h, op].service_is_ready()}
        deadline = time.monotonic() + CALL_TIMEOUT_S
        while not all(f.done() for f in futures.values()) and time.monotonic() < deadline:
            time.sleep(0.01)
        return {h: futures[h].result() if h in futures and futures[h].done()
                else Trigger.Response(success=False, message=f"no reply from /gello/{h}/{op}: is it running?")
                for h in HANDS}

    def _on_arm(self, _request, response: Trigger.Response) -> Trigger.Response:
        replies = self._call("arm")
        response.success = all(r.success for r in replies.values())
        response.message = "; ".join(f"{h}: {r.message}" for h, r in replies.items())
        if response.success:
            self.get_logger().info(response.message)
        else:
            if any(r.success for r in replies.values()):
                self._call("disarm")
                response.message += "; disarmed both again, so neither arm follows"
            self.get_logger().warning(response.message)
        return response

    def _on_disarm(self, _request, response: Trigger.Response) -> Trigger.Response:
        replies = self._call("disarm")
        response.success = all(r.success for r in replies.values())
        response.message = "; ".join(f"{h}: {r.message}" for h, r in replies.items())
        return response

    def close(self) -> None:
        pass


def main() -> None:
    run(GelloPair)


if __name__ == "__main__":
    main()
