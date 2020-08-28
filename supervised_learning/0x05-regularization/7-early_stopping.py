#!/usr/bin/env python3
"""early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """destermines if GD should be stopped early"""

    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        return True, count
    else:
        return False, count
