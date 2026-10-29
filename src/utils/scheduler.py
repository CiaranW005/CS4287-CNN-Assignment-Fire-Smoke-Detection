import numpy as np

T_max = 8 #FIXME find a way not ot hardcode this
min_factor = 0.1

def cosine_decay(epoch):
    progress = min(epoch / T_max, 1.0)

    return min_factor + (1 - min_factor) * (
        1 + np.cos(np.pi * progress)
    ) / 2
