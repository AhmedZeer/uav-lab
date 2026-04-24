#!/usr/bin/env bash
set -euo pipefail

${ISAAC_LAB_DIR}/isaaclab.sh -p scripts/spawn_uav_ui.py \
    --ui \
    --follow_cam \
    --follow_cam_smooth_tau 0.1 \
    --follow_cam_distance 15 \
    --follow_cam_env_id 10 \
    --start_alt 100 \
    --start_speed 100 \
    --thrust 0.5 \
    --roll -0.00 \
    --pitch -0.0 \
    --propeller_anim \
    --num_envs 10 \
    --env_spacing 5 \
    --start_roll_noise_deg 1.2 \
    --start_pitch_noise_deg 2.5 \
    --start_yaw_noise_deg 0.0 \
    --start_speed_noise_mps 0.0 \
    --start_body_rate_noise_rps 0.00
