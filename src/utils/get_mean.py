
def mean_valid(values):
    valid = values[values >= 0]
    return valid.mean().item() if valid.numel() else float("nan")