import numpy as np

r = np.load('./heatmaps_npy/heatmaps_00000001_00000001.npy');

print(r.ndim, r.shape, r.dtype, r.itemsize) # 3 (95, 511, 12) float32 4

print(r[:,:,0])
print(r[:,:,0].shape)