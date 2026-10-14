# rollout

Runs a policy exactly like `lerobot-rollout`, but lets you change its task,
pause it or send the arms home while it runs — without reloading the model
(~90 s).

```
live.py   the rollout loop + the /rollout/* ROS interface
```

Works the same on the real arms and the Isaac twin.

## Using it

```bash
./scripts/rollout.sh --mode=live

ros2 topic pub --once /rollout/task std_msgs/msg/String "{data: 'pick up the scissors'}"
ros2 service call /rollout/reset std_srvs/srv/Trigger
ros2 topic echo /rollout/status
```

Or use the web page: [`../harness/`](../harness/README.md).

| name | type | |
|---|---|---|
| `/rollout/task` | topic in | new task for the policy |
| `/rollout/reset` | service | send arms home, clear queued actions, wait for a new task |
| `/rollout/home` | service | send arms home and wait |
| `/rollout/park` | service | park the arms and wait |
| `/rollout/pause`, `/rollout/resume` | service | stop / continue stepping the policy |
| `/rollout/quit` | service | end the rollout |
| `/rollout/status` | topic out | JSON: task, paused, held, ticks, resets, cameras (2 Hz) |
| `/rollout/state`, `/rollout/images/<cam>/compressed` | topic out | what the policy was shown |

## Behaviour

- **A new task takes effect immediately** — queued actions from the old task
  are thrown away.
- **Commands are queued.** A service answers right away; the loop applies it
  between ticks. A reset is finished when `resets` in the status goes up.
- **`reset`, `home` and `park` leave the robot holding still** (`held: true`)
  until you send a task or `resume`. `pause` is different: the task stays,
  stepping just stops.
- **Going home is a smooth 5 s move**, not a jump. `--home-duration` changes it;
  `--home` changes the target (default all zeros).
- State and images are published every `--publish-every` ticks (`rollout.sh`
  uses 5) to keep the control loop fast.
