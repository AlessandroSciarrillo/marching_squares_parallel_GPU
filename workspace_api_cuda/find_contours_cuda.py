from cuda import cuda, nvrtc
import numpy as np

def _get_contour_segments():
    k=0
    for i in range(1000):
        for j in range(10000):
            k=i*j

    return k