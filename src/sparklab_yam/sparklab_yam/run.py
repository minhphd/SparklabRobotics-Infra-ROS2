"""``main()`` for nodes that must still talk to ROS while shutting down."""

from __future__ import annotations

import signal

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions


def run(node_cls) -> None:
    """Spin a node until SIGINT/SIGTERM, then call its ``close()``.

    rclpy's own handler would shut the context down first, leaving ``close()``
    unable to publish. Further signals are ignored during ``close()`` so a second
    Ctrl-C cannot cut a park short. See DESIGN.md#ros-2.
    """
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # A background job inherits SIGINT ignored; restore it with SIGTERM.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = node_cls()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        # A timeout so the KeyboardInterrupt is raised promptly.
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        executor.shutdown()
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
