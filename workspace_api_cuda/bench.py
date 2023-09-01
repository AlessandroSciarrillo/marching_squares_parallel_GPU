#**********************************************#
#
#   Benchmark API Cuda Version
#
#**********************************************#

import time
import numpy as np
import matplotlib.pyplot as plt #WARNING: se viene rimosso Cuda genera Illegal Access Memory
from cuda import cuda, nvrtc, cudart # cudart only for GPUs info

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
times = 1000
inputReal = False

if (inputReal):
    # Get a Real Input
    t = np.load('./heatmaps_00000001_00000001.npy');
    r = t[:,:,10];
else:
    # Construct artificial test data
    x, y = np.ogrid[-np.pi:np.pi:95j, -np.pi:np.pi:511j]
    r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))


# Load Cuda kernel
kernel, bufferSize, stream, args, \
result_1x, result_1y, result_2x, result_2y, \
dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, \
dImageclass, \
NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, \
module, context = load_kernel(r.size, r.shape[1], r.shape[0], 0.5)

print(r.shape)

# Print some benchmark and GPUs info
print(" ______________________________________________________________________________")
print("|__________________________________BENCHMARK___________________________________|")
print("|")
print("| INFO:")
print("|    Input type         => ", "real" if(inputReal) else "artificial")
print("|    Number of launches => ", times)
print("|")
err, nDevices = cuda.cuDeviceGetCount()
print("| GPUs [", nDevices,"]:" )
for device in range(nDevices):
    err, prop = cudart.cudaGetDeviceProperties(device)
    print("|    Device number                => ", device)
    print("|    Device name                  => ", prop.name.decode("utf-8"))
    print("|    Memory Clock Rate (KHz)      => ", prop.memoryClockRate)
    print("|    Memory Bus Width (bits)      => ", prop.memoryBusWidth)
    print("|    Peak Memory Bandwidth (GB/s) => ", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6)
    print("|")
print("|______________________________________________________________________________\n")


# Take skimage lib times
print("Launch skimage version bench...  ", end = '')
st = time.time()
for a in range(times):

    #placeholder
    contours = measure.find_contours(r, 0.5)

et = time.time()
elapsed_time_lib = (et - st)/times #TODO bisogna prendere solo il tempo di esecuzione del cython escudeldo la chiamata ad assembly_contours
print("[" + u'\u2713' + "]")


# Take my version times
print("Launch API CUDA version bench... ", end = '')
st = time.time()
for a in range(times):
    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    # Only kernel execution
    segments, _ = launch_kernel( kernel, bufferSize, stream, args,
                                    result_1x, result_1y, result_2x, result_2y, 
                                    dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, dImageclass, 
                                    NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, 
                                    r.astype(np.float64), 0.5)
et = time.time()
elapsed_time_my = (et - st)/times
print("[" + u'\u2713' + "]\n")

# Free Cuda Kernel memory
err, = cuda.cuStreamDestroy(stream)
err, = cuda.cuMemFree(dImageclass)
err, = cuda.cuMemFree(dResult1Xclass)
err, = cuda.cuMemFree(dResult1Yclass)
err, = cuda.cuMemFree(dResult2Xclass)
err, = cuda.cuMemFree(dResult2Yclass)
err, = cuda.cuModuleUnload(module) 
err, = cuda.cuCtxDestroy(context)  

# Print times, difference and speedup
print('RESULTS:')
print('MS execution time lib:   ', elapsed_time_lib, 'seconds   WARNING: full time with the ending data recostruction!')
print('MS execution time my:    ', elapsed_time_my, 'seconds')
print('Difference:              ', elapsed_time_lib - elapsed_time_my, 'seconds')
print('Speedup:                 ', elapsed_time_lib / elapsed_time_my , 'seconds\n')


