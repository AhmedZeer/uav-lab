#!/usr/bin/env bash
set -euo pipefail

${ISAAC_LAB_DIR}/isaaclab.sh -p scripts/spawn_uav_surfaces_ui.py \
    --ui \
    --follow_cam \
    --follow_cam_smooth_tau 0.1 \
    --start_alt 100 \
    --start_speed 100 \
    --thrust 0.5 \
    --roll -0.00 \
    --pitch -0.0 \
    --propeller_anim