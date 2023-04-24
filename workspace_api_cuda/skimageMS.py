import numpy as np
import matplotlib.pyplot as plt
import time

from skimage import measure
from _find_contours import find_contours_full as my_find_contours_full
from _find_contours import find_contours_splitted as my_find_contours_splitted


# Construct some test data
#x, y = np.ogrid[-np.pi:np.pi:95j, -np.pi:np.pi:511j]
#r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
t = np.load('./heatmaps_00000001_00000001.npy');
r = t[:,:,10];

times = 1000;

st = time.time()
################ LOAD CUDA KERNEL ######################
from load_cuda_kernel import load_kernel

kernel, bufferSize, stream, args, \
result_1x, result_1y, result_2x, result_2y, \
dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, \
dImageclass, \
NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, \
module, context = load_kernel(r.size, r.shape[1], r.shape[0], 0.5)
#######################################################
et = time.time()
elapsed_time_cuda_pre = et - st
print('Compilazione Kernel Cuda:', elapsed_time_cuda_pre, 'seconds')

# Print Diff with lib
st = time.time()
for a in range(times):
    contours = measure.find_contours(r, 0.5)  # skimage
et = time.time()
elapsed_time_lib = (et - st)/times
print('Execution time lib :', elapsed_time_lib, 'seconds')


st = time.time()

for a in range(times):
    #contours = my_find_contours_full(r,0.5)

    contours = my_find_contours_splitted(
        kernel, bufferSize, stream, args,
        result_1x, result_1y, result_2x, result_2y,
        dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass,
        dImageclass,
        NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y,
        r, 0.5) 

et = time.time()
elapsed_time_my = (et - st)/times
print('Execution time cuda:', elapsed_time_my, 'seconds')
print('Difference :', elapsed_time_my - elapsed_time_lib, 'seconds')
print('Speedup :', elapsed_time_lib / elapsed_time_my , 'seconds')


################ CLEAN CUDA KERNEL ######################
from cuda import cuda, nvrtc

err, = cuda.cuStreamDestroy(stream)
err, = cuda.cuMemFree(dImageclass)
err, = cuda.cuMemFree(dResult1Xclass)
err, = cuda.cuMemFree(dResult1Yclass)
err, = cuda.cuMemFree(dResult2Xclass)
err, = cuda.cuMemFree(dResult2Yclass)
err, = cuda.cuModuleUnload(module) 
err, = cuda.cuCtxDestroy(context)  
#######################################################


# Display the image and plot all contours found
print(r.size, r.shape)
fig, ax = plt.subplots()
ax.imshow(r, cmap=plt.cm.gray)

#print(contours)
for contour in contours:
    if 90 < max(contour[:, 1]) < 2000:
        #print(max(contour[:, 0]))
        pos = np.argmax(contour[:, 1])
        #print(contour[pos,0], contour[pos,1])
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
