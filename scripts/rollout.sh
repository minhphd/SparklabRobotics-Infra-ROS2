#!/usr/bin/env bash
# Run a policy. Robot, policy and strategy are three independent choices.
#
#     ./scripts/rollout.sh                              # real arms, fine-tuned
#     ./scripts/rollout.sh --policy=stock-typed         # stock baseline
#     ./scripts/rollout.sh --mode=dagger                # teleop takeover
#     ./scripts/rollout.sh --robot=sim --policy=stock   # Isaac twin
#     ./scripts/rollout.sh --mode=live                  # retargetable while running
#     ./scripts/rollout.sh --dry-run ...                # print the command only
#
# Unrecognised flags pass through to the runner, so any lerobot-rollout flag
# works without editing this file:
#
#     ./scripts/rollout.sh --duration=120 --policy.num_flow_timesteps=4
#
# Env knobs: FINETUNED_PATH STOCK_REPO FOXGLOVE_PORT CPU_CORES PUBLISH_EVERY
#            DAGGER_EPISODES DAGGER_REPO_ID DAGGER_RECORD_AUTONOMOUS
#            ROLLOUT_SKIP_PREFLIGHT

set -euo pipefail

# Checkpoint paths are relative to the repo root (../SparkRobot/...), so anchor
# there rather than to wherever this was called from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FINETUNED_PATH="${FINETUNED_PATH:-../SparkRobot/MolmoAct2/lerobot_training/last/pretrained_model}"
STOCK_REPO="${STOCK_REPO:-lerobot/MolmoAct2-BimanualYAM-LeRobot}"
FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"

ROBOT=hw
POLICY=finetuned
MODE=base
TASK="put the marker into the cardboard box"
DURATION=""          # resolved after parsing; the default depends on the mode
USE_FOXGLOVE=1
DRY_RUN=0
PASSTHRU=()

usage() {
    cat <<'EOF'
usage: rollout.sh [--robot=hw|sim] [--policy=finetuned|stock|stock-typed]
                  [--mode=base|dagger|live] [--task=STR] [--duration=N]
                  [--no-foxglove] [--dry-run] [extra lerobot-rollout flags...]

  --robot    hw          real arms, via the sparklab_yam drivers (default)
             sim         the Isaac twin, via sparklab_sim.policy_server

  --policy   finetuned   local fine-tune at $FINETUNED_PATH (default)
             stock       stock checkpoint as a LeRobot policy dir; gets the
                         camera-key rename its input_features need
             stock-typed stock checkpoint configured by --policy.type, with
                         every setup flag spelled out and no rename. Allows
                         network on the first run so it can download.

  --mode     base        autonomous rollout (default)
             dagger      human-in-the-loop correction capture; hw only, Quest
                         takeover through the bridge
             live        /rollout/* ROS channel for retargeting mid-run, driven
                         from `ros2 topic pub` / `ros2 service call` or harness.sh

env: FINETUNED_PATH STOCK_REPO FOXGLOVE_PORT DAGGER_REPO_ID PUBLISH_EVERY
     ROLLOUT_SKIP_PREFLIGHT=1  skip the "is it up" checks
EOF
}

die() { printf 'rollout.sh: %s\n' "$1" >&2; exit 2; }

# --- arguments --------------------------------------------------------------
# `--task x` as well as `--task=x`, or a multi-word task splits into PASSTHRU.
while [ $# -gt 0 ]; do
    arg="$1"; val="${arg#*=}"
    case "$arg" in
        --robot|--policy|--mode|--task|--duration)
            [ $# -ge 2 ] || die "$arg needs a value"
            val="$2"; shift ;;
    esac
    case "$arg" in
        --robot|--robot=*)       ROBOT="$val" ;;
        --policy|--policy=*)     POLICY="$val" ;;
        --mode|--mode=*)         MODE="$val" ;;
        --task|--task=*)         TASK="$val" ;;
        --duration|--duration=*) DURATION="$val" ;;
        --no-foxglove)           USE_FOXGLOVE=0 ;;
        --dry-run)               DRY_RUN=1 ;;
        -h|--help)               usage; exit 0 ;;
        *)                       PASSTHRU+=("$arg") ;;
    esac
    shift
done

case "$ROBOT"  in hw|sim) ;;                      *) die "unknown --robot=$ROBOT (hw|sim)" ;; esac
case "$POLICY" in finetuned|stock|stock-typed) ;; *) die "unknown --policy=$POLICY (finetuned|stock|stock-typed)" ;; esac
case "$MODE"   in base|dagger|live) ;;            *) die "unknown --mode=$MODE (base|dagger|live)" ;; esac

# dagger needs a human taking over through the Quest bridge onto real arms.
if [ "$MODE" = dagger ] && [ "$ROBOT" != hw ]; then
    die "--mode=dagger requires --robot=hw"
fi

# dagger is bounded by --strategy.num_episodes; a wall-clock limit would cut a
# session off mid-correction. lerobot-rollout treats 0 as infinite.
if [ -z "$DURATION" ]; then
    case "$MODE" in dagger) DURATION=0 ;; *) DURATION=2000 ;; esac
fi

# --- preflight --------------------------------------------------------------
# Otherwise a missing dependency is an opaque timeout after the model load.

port_open() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]"
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&- 3>&-
    fi
}

# require_topics <topic>[,<topic>...] <what> <how-to-start>
require_topics() {
    if [ "${ROLLOUT_SKIP_PREFLIGHT:-0}" = "1" ]; then return 0; fi
    command -v ros2 >/dev/null || die "ros2 not found — source install/setup.bash"
    local topics="$1" what="$2" how="$3" listed missing=()
    listed=$(ros2 topic list 2>/dev/null)
    local IFS=','
    for t in $topics; do grep -qx "$t" <<<"$listed" || missing+=("$t"); done
    unset IFS
    if [ ${#missing[@]} -eq 0 ]; then
        printf '[preflight] %-14s ... OK\n' "$what" >&2
        return 0
    fi
    printf '[preflight] %-14s ... MISSING %s\n' "$what" "${missing[*]}" >&2
    printf '            start it with: %s\n' "$how" >&2
    printf '            (ROLLOUT_SKIP_PREFLIGHT=1 to bypass)\n' >&2
    exit 1
}

# require_port <port>[,<port>...] <what> <how-to-start>
require_port() {
    if [ "${ROLLOUT_SKIP_PREFLIGHT:-0}" = "1" ]; then return 0; fi
    local ports="$1" what="$2" how="$3" missing=()
    local IFS=','
    for p in $ports; do port_open "$p" || missing+=("$p"); done
    unset IFS
    if [ ${#missing[@]} -eq 0 ]; then
        printf '[preflight] %-14s %-12s ... OK\n' "$what" "$ports" >&2
        return 0
    fi
    printf '[preflight] %-14s %-12s ... NOT LISTENING (%s)\n' "$what" "$ports" "${missing[*]}" >&2
    printf '            start it with: %s\n' "$how" >&2
    printf '            (ROLLOUT_SKIP_PREFLIGHT=1 to bypass)\n' >&2
    exit 1
}

# --- robot ------------------------------------------------------------------
ROBOT_ARGS=()
PIN=()

if [ "$ROBOT" = hw ]; then
    require_topics "/left/joint_states,/right/joint_states" "rig" \
        "ros2 launch sparklab_yam yam.launch.py"
    if [ "$MODE" = dagger ]; then
        require_topics "/quest/right/pose" "quest bridge" "ros2 launch sparklab_yam yam.launch.py"
    fi

    ROBOT_ARGS=( --robot.type=yam_ultra_bimanual )

    cat >&2 <<'EOF'
[hw] The drivers own the CAN loop and the torque. Do not bring the rig up with
     teleop:=true alongside a rollout — that is two control loops on the arms.
EOF

    # huggingface_hub still makes a network round trip per file to check etags
    # against the cache, mid-run; offline skips it.
    export HF_HUB_OFFLINE=1

    # CPU pin, off by default: it starved the loop it was meant to protect —
    # 30 Hz unpinned against ~4 Hz under `taskset -c 0-5`, same checkpoint.
    if [ -n "${CPU_CORES-}" ]; then PIN=( taskset -c "$CPU_CORES" ); fi
else
    require_port "8081" "sim policy" \
        "./scripts/isaac_python.sh -m sparklab_sim.policy_server --port 8081 --physics"
    ROBOT_ARGS=( --robot.type=yam_ultra_sim )
fi

# --- policy -----------------------------------------------------------------
# Maps the checkpoint's camera keys onto the rig's, keyed off the policy.
RENAME=(
    --rename_map='{"observation.images.top": "observation.images.top", "observation.images.right": "observation.images.right_wrist", "observation.images.left": "observation.images.left_wrist"}'
)

POLICY_ARGS=()
RENAME_ARGS=()

case "$POLICY" in
    # A --policy.path load gets nothing added: config.json owns dtype, chunking,
    # cuda graphs and amp. Never --policy.use_amp=true — fp16 over bfloat16.
    finetuned)
        POLICY_ARGS=( --policy.path="${FINETUNED_PATH}" )
        RENAME_ARGS=( "${RENAME[@]}" )
        ;;
    stock)
        POLICY_ARGS=( --policy.path="${STOCK_REPO}" )
        RENAME_ARGS=( "${RENAME[@]}" )
        ;;
    # Configured by --policy.type, so every flag is spelled out. image_keys
    # names the wrist keys directly, so this variant must NOT get the rename.
    stock-typed)
        POLICY_ARGS=(
            --policy.type=molmoact2
            --policy.checkpoint_path="${STOCK_REPO}"
            --policy.action_mode=continuous
            --policy.setup_type="bimanual yam robotic arms in molmoact2"
            --policy.control_mode="absolute joint pose"
            --policy.normalize_gripper=false
            --policy.num_flow_timesteps=8
            --policy.image_keys='["observation.images.top","observation.images.left_wrist","observation.images.right_wrist"]'
            --policy.inference_action_mode=continuous
            --policy.device=cuda
            --policy.model_dtype=bfloat16
            --policy.enable_inference_cuda_graph=true
            --policy.n_action_steps=30
            --policy.chunk_size=30
        )
        # The baseline you reach for when it may not be cached yet, where
        # offline would turn a download into "not found in cache".
        unset HF_HUB_OFFLINE
        ;;
esac

# --- strategy ---------------------------------------------------------------
STRATEGY_ARGS=( --strategy.type=base )

if [ "$MODE" = dagger ]; then
    STRATEGY_ARGS=(
        --strategy.type=dagger
        --strategy.input_device=keyboard
        # false records only the correction windows, one episode each; true is
        # continuous capture — more volume, less signal.
        --strategy.record_autonomous="${DAGGER_RECORD_AUTONOMOUS:-false}"
        --strategy.num_episodes="${DAGGER_EPISODES:-20}"
        --teleop.type=bi_quest_teleop
        # A new repo_id: DAgger tags every frame with an `intervention` bool
        # that put-box's schema lacks, so it cannot be appended to.
        --dataset.repo_id="${DAGGER_REPO_ID:-minhphd/put-box-dagger}"
        --dataset.single_task="${TASK}"
        --dataset.fps=30
        --dataset.push_to_hub=false
    )
    cat >&2 <<'EOF'
[dagger] keyboard: p pause/resume | c toggle correction recording (while
         paused) | u push to hub | ESC stop. No foot pedal attached, so this
         needs a second person at the keyboard while you are in the headset.
EOF
fi

# --- runner -----------------------------------------------------------------
if [ "$MODE" = live ]; then
    # Wraps the rollout in the /rollout/* channel harness.sh drives.
    export HF_HUB_OFFLINE=1
    CMD=( "${PIN[@]}" ros2 run lerobot_robot_sparklab rollout
          --publish-every "${PUBLISH_EVERY:-5}" )
    # Sim-tuned. On hardware, leave LeRobot's defaults and pass --fps through.
    if [ "$ROBOT" = sim ]; then
        CMD+=( --interpolation_multiplier 2 --fps 20 )
    fi
    printf '[live] /rollout/* comes up after the model loads — attach ./scripts/harness.sh,\n' >&2
    printf '[live] or e.g. ros2 service call /rollout/reset std_srvs/srv/Trigger\n' >&2
else
    CMD=( "${PIN[@]}" lerobot-rollout )
fi

# Foxglove ABORTS on a bound port rather than falling back, so a second
# concurrent rollout dies at startup unless you move FOXGLOVE_PORT.
FOXGLOVE_ARGS=()
if [ "$USE_FOXGLOVE" = 1 ]; then
    FOXGLOVE_ARGS=(
        --display_data true
        --display_mode=foxglove
        --display_ip=127.0.0.1
        --display_port="$FOXGLOVE_PORT"
    )
fi

CMD+=( "${STRATEGY_ARGS[@]}" "${POLICY_ARGS[@]}" "${ROBOT_ARGS[@]}"
       --task="${TASK}" "${RENAME_ARGS[@]}" "${FOXGLOVE_ARGS[@]}"
       --duration="${DURATION}" "${PASSTHRU[@]}" )

printf '[rollout] robot=%s policy=%s mode=%s duration=%s\n' \
    "$ROBOT" "$POLICY" "$MODE" "$DURATION" >&2

if [ "$DRY_RUN" = 1 ]; then
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi
exec "${CMD[@]}"
