#!/usr/bin/env python3
"""Export aggregate success counts using the shared publication renderer."""
import argparse
from pathlib import Path
import matplotlib
from g1_analysis.paper_plots import (load_declared_success_counts, success_plot, PUBLICATION_STYLE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    counts = load_declared_success_counts(args.counts)
    with matplotlib.rc_context(PUBLICATION_STYLE):
        success_plot([], args.output, declared_counts=counts)


if __name__ == '__main__':
    main()
