#!/usr/bin/env python3
"""Single CLI entry point. Run from a sourced ROS 2 environment for bag reading."""
from g1_analysis.cli import main

if __name__ == '__main__':
    raise SystemExit(main())

"""
python3 main.py \
  --root "$HOME/g1_ws/experiments" \
  --annotations config/navigation_annotations.csv \
  --success-counts config/success_counts.csv \
  --output "$HOME/g1_ws/analysis/navigation_v2_full"

python3 main.py \
  --root "$HOME/g1_ws/experiments" \
  --annotations config/navigation_annotations.csv \
  --output "$HOME/g1_ws/analysis/navigation_v2_full"
  
python3 main.py \
  --results "$HOME/g1_ws/analysis/navigation_v2_full" \
  --output "$HOME/g1_ws/analysis/navigation_v2_replot" \
  --annotations config/navigation_annotations.csv

"""