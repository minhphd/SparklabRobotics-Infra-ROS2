import importlib.util
import os

# cap-x's model client loads the cap-x checkout's .env (API keys) on import; tests need none.
os.environ.setdefault("CAPX_SKIP_DOTENV", "1")

# Without cap-x's stack (scripts/capx_python.sh) don't import these at all: a module-level
# skip inside ROS's launch_testing plugin skips the whole session, every package's tests.
if importlib.util.find_spec("curobo") is None:
    collect_ignore = ["test_robot.py", "test_env.py", "test_api.py", "test_config.py"]
