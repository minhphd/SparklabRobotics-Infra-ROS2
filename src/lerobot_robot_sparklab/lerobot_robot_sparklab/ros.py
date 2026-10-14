"""rclpy plumbing for LeRobot processes, which own the main thread and signals."""

from __future__ import annotations

import atexit
import threading
from collections.abc import Callable

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


def init() -> None:
    """Initialise rclpy once per process, leaving SIGINT to LeRobot.

    rclpy's own handler would shut the context down on Ctrl-C, before
    ``disconnect()`` can park the arms over ROS. See DESIGN.md#ros-2.
    """
    if not rclpy.ok():
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)


def spin_in_background(node: Node) -> Callable[[], None]:
    """Spin ``node`` on a daemon thread. Returns a function that stops and destroys it.

    Stopping joins the thread before tearing the executor down: Humble's
    ``Executor.shutdown()`` aborts the process if a spin is still waiting.
    """
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stop = threading.Event()

    def spin() -> None:
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    thread = threading.Thread(target=spin, daemon=True, name=node.get_name())
    thread.start()

    def close() -> None:
        if stop.is_set():
            return
        stop.set()
        thread.join()
        executor.shutdown()
        node.destroy_node()

    # Also on exit: a process that dies with a spinner running aborts in rclpy.
    atexit.register(close)
    return close
