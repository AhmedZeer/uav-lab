#!/usr/bin/env bash
set -euo pipefail

${ISAAC_LAB_DIR}/isaaclab.sh -p scripts/spawn_uav_surfaces_ui.py \
    --ui \
    --telemetry_ui \
    --follow_cam \
    --follow_cam_smooth_tau 0.1 \
    --start_alt 100 \
    --start_speed 40 \
    --log_csv \
    --log_frame_debug \
    --log_surface_debug \
    --log_path logs/airborne_trim.csv
