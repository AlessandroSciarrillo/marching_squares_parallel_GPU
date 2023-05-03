#**********************************************#
#
#   Benchmark API Cuda Version
#
#**********************************************#

import time
import numpy as np
from cuda import cuda, nvrtc

from load_cuda_kernel import load_kernel
from launch_cuda_kernel import launch_kernel

from skimage import measure
from _find_contours import find_contours_splitted as my_find_contours_splitted

# For API Cuda error check
def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            print("\nError string: ",cuda.cuGetErrorString(err),"\n")
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

# Set Benchmark parameters
times = 1000;

# Get a Real Input
t = np.load('./heatmaps_00000001_00000001.npy');
r = t[:,:,10];


# Load Cuda kernel
kernel, bufferSize, stream, args, \
result_1x, result_1y, result_2x, result_2y, \
dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, \
dImageclass, \
NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, \
module, context = load_kernel(r.size, r.shape[1], r.shape[0], 0.5)


# Take skimage lib times
st = time.time()

#

et = time.time()
elapsed_time_lib = (et - st)/times


# Take my version times
st = time.time()

for a in range(times):
    print("launch: ",a)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    # segments, _ = launch_kernel( kernel, bufferSize, stream, args,
    #                                 result_1x, result_1y, result_2x, result_2y, 
    #                                 dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, dImageclass, 
    #                                 NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, 
    #                                 r.astype(np.float64), float(0.5))
    contours, elapsed_time_assemble_con, elapsed_time_kernel = my_find_contours_splitted(
        kernel, bufferSize, stream, args,
        result_1x, result_1y, result_2x, result_2y,
        dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass,
        dImageclass,
        NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y,
        r, 0.5) 

et = time.time()
elapsed_time_my = (et - st)/times


# Print times, difference and speedup
print('MS execution time lib :', elapsed_time_lib, 'seconds')
print('MS execution time my  :', elapsed_time_my, 'seconds')
print('Difference :', time_lib_MS - time_my_MS, 'seconds')
print('Speedup :', time_lib_MS / time_my_MS , 'seconds')


# Free Cuda Kernel memory
err, = cuda.cuStreamDestroy(stream)
err, = cuda.cuMemFree(dImageclass)
err, = cuda.cuMemFree(dResult1Xclass)
err, = cuda.cuMemFree(dResult1Yclass)
err, = cuda.cuMemFree(dResult2Xclass)
err, = cuda.cuMemFree(dResult2Yclass)
err, = cuda.cuModuleUnload(module) 
err, = cuda.cuCtxDestroy(context)  