"""Evaluation harness — MOS correlation for UPIQAL.

Run `python3 eval/mos_correlation.py --dataset kadid10k` to get
SROCC / PLCC / RMSE numbers for the current pipeline on KADID-10k.

This is the canonical way to compare pipeline revisions objectively.
The 3-image `self/diff/cartoon` triple in the README is a smoke test,
not a benchmark.
"""
