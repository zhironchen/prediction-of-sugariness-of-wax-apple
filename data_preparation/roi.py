import math

def roi(data_temp, row1, row2, col1, col2):
    x, y, z = data_temp.shape
    roi = data_temp[math.ceil(x / 2) + row1:math.ceil(x / 2) + row2,
           math.ceil(y / 2) + col1:math.ceil(y / 2) + col2]
    return roi
