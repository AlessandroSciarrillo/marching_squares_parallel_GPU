#************************************************************#
#
#   Benchmark API Cuda Marching Squares Complete Version 
#
#************************************************************#

import time
import numpy as np
import matplotlib.pyplot as plt #WARNING: se viene rimosso Cuda genera Illegal Access Memory
from cuda import cuda, nvrtc, cudart # cudart only for GPUs info

from skimage import measure
from full_MS import bench_marching_squares_gpu

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
times = 50000
inputReal = False

if (inputReal):
    # Get a Real Input
    t = np.load('./heatmaps_00000001_00000001.npy');
    image = t[:,:,10];
else:
    # Construct artificial test data
    x, y = np.ogrid[-np.pi:np.pi:95j, -np.pi:np.pi:511j]
    image = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Print some benchmark and GPUs info
print(" ______________________________________________________________________________")
print("|__________________________________BENCHMARK___________________________________|")
print("|")
print("| INFO:")
print("|    Input type         => ", "real" if(inputReal) else "artificial")
print("|    Image shape        => ", image.shape)
print("|    Number of launches => ", times)
print("|______________________________________________________________________________\n")


# Take skimage lib times
print("Launch skimage version bench...  ", end = '')
st = time.time()
for a in range(times):

    #placeholder
    contours = measure.find_contours(image, 0.5)

et = time.time()
elapsed_time_lib = (et - st)/times #TODO bisogna prendere solo il tempo di esecuzione del cython escudeldo la chiamata ad assembly_contours
print("[" + u'\u2713' + "]")


# Take my version times
print("Launch API CUDA version bench... \n", end = '')

elapsed_time_my = bench_marching_squares_gpu(image, times)

print("[" + u'\u2713' + "]\n")


# Print times, difference and speedup
print('RESULTS:')
print('MS execution time lib:   ', elapsed_time_lib, 'seconds   WARNING: full time with the ending data recostruction!')
print('MS execution time my:    ', elapsed_time_my, 'seconds')
print('Difference:              ', elapsed_time_lib - elapsed_time_my, 'seconds')
print('Speedup:                 ', elapsed_time_lib / elapsed_time_my , 'seconds\n')


