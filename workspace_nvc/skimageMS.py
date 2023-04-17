import numpy as np
#import matplotlib.pyplot as plt
import time

from skimage import measure
from _find_contours import find_contours as my_fc

# Construct some test data
#x, y = np.ogrid[-np.pi:np.pi:950j, -np.pi:np.pi:511j]
#r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
t = np.load('./heatmaps_00000001_00000001.npy');
r = t[:,:,8];

st = time.time()
for c in range(11):
    r=t[:,:,c]
    contours = measure.find_contours(r, 0.5)
et = time.time()
elapsed_time_lib = et - st
print('Lib Execution times:', elapsed_time_lib, 'seconds')

st = time.time()
for c in range(11):
    r=t[:,:,c]
    contours = measure.find_contours(r, 0.5)
et = time.time()
elapsed_time_lib2 = et - st
print('Lib2 Execution times:', elapsed_time_lib2, 'seconds')

st = time.time()
for c in range(11):
    r=t[:,:,c]
    contours = my_fc(r, 0.5) 
et = time.time()
elapsed_time_my = et - st
print('nvc Execution times:', elapsed_time_my, 'seconds')

st = time.time()
for c in range(11):
    r=t[:,:,c]
    contours = my_fc(r, 0.5) 
et = time.time()
elapsed_time_my2 = et - st
print('nvc2 Execution times:', elapsed_time_my2, 'seconds')

print('Difference :', elapsed_time_my2 - elapsed_time_lib2, 'seconds')
print('Speedup :', elapsed_time_lib2 / elapsed_time_my2 , 'seconds')

