#!/usr/bin/env bash
set -euo pipefail

${ISAAC_LAB_DIR}/isaaclab.sh -p scripts/spawn_uav_ui.py \
    --ui \
    --follow_cam \
    --follow_cam_smooth_tau 0.1 \
    --start_alt 100 \
    --start_speed 100 \
    --thrust 0.2 \
    --roll -0.00 \
    --pitch -0.0 \
    --propeller_anim \
    --num_envs 1 \


# ${ISAAC_LAB_DIR}/isaaclab.sh -p scripts/spawn_uav_ui.py \
#     --ui \
#     --follow_cam \
#     --follow_cam_smooth_tau 0.1 \
#     --start_alt 2 \
#     --start_speed 0 \
#     --thrust 0.0 \
#     --roll 0.00 \
#     --pitch 0.0 \
#     --yaw 0.0 \
#     --propeller_anim \
#     --num_envs 1 \
