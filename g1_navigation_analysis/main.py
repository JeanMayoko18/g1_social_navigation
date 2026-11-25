#!/usr/bin/env python3
"""Single CLI entry point. Run from a sourced ROS 2 environment for bag reading."""
from g1_analysis.cli import main

if __name__ == '__main__':
    raise SystemExit(main())

"""
python3 main.py \
  --root "$HOME/g1_ws/experiments" \
  --output "$HOME/g1_ws/analysis/navigation_inventory" \
  --inventory-only


source /opt/ros/humble/setup.bash

cd "$HOME/g1_ws/scripts/g1_navigation_analysis"

python3 main.py \
  --root "$HOME/g1_ws/experiments" \
  --output "$HOME/g1_ws/analysis/navigation_all_v1"

"""