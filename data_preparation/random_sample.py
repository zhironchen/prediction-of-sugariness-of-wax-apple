import random
import numpy as np

def data_sampling_random(x, y, num, keep_left=True):
    # Random sampling of data. This function will return the data selected by algorithm, set keep_left = True to return both of the selected and not seltected data.
    random.seed(42)
    tatal_sample_list = list(range(0, len(x)))
    selected_index = random.sample(tatal_sample_list, num)
    x_new = []
    y_new = []
    x_left = []
    y_left = []
    for i in tatal_sample_list:
        if i in selected_index:
            x_new.append(x[i])
            y_new.append(y[i])
        else:
            x_left.append(x[i])
            y_left.append(y[i])
    return (x_new, y_new, x_left, y_left) if keep_left else (x_new, y_new)
