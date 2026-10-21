"""Entry point for capx_python.sh: cap-x's site-packages first, then ROS 2's, then run like ``python``."""

import code
import os
import runpy
import site
import sys

# cap-x imports LIBERO, which asks on stdin for a dataset folder the first time an
# account imports it; write its defaults for this checkout instead, every run.
libero = os.environ.setdefault(
    "LIBERO_CONFIG_PATH", os.path.join(os.path.expanduser("~"), ".cache", "capx_sparklab", "libero"))
if libero == os.path.join(os.path.expanduser("~"), ".cache", "capx_sparklab", "libero"):
    root = os.path.join(os.environ["CAPX"], "capx", "third_party", "LIBERO-PRO", "libero", "libero")
    os.makedirs(libero, exist_ok=True)
    with open(os.path.join(libero, "config.yaml"), "w") as f:
        for key, path in (("benchmark_root", ""), ("bddl_files", "bddl_files"), ("init_states", "init_files"),
                          ("datasets", "../datasets"), ("assets", "assets")):
            f.write(f"{key}: {os.path.normpath(os.path.join(root, path))}\n")

# Ahead of ROS 2 so cap-x's numpy 1.26 and CUDA torch win over sparklab-ros's.
site.addsitedir(os.environ["CAPX_SITE"])
sys.path += [p for p in os.environ["ROS_PYTHONPATH"].split(os.pathsep) if p and p not in sys.path]

args = sys.argv[1:]
if args[:1] == ["-m"] and len(args) > 1:
    sys.argv = args[1:]
    runpy.run_module(args[1], run_name="__main__", alter_sys=True)
elif args:
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
else:
    code.interact()
