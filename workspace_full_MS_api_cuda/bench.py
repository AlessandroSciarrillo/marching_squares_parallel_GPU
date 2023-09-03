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

from collections import deque
def _assemble_contours(segments):
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in segments:
        # Ignore degenerate segments.
        # This happens when (and only when) one vertex of the square is
        # exactly the contour level, and the rest are above or below.
        # This degenerate vertex will be picked up later by neighboring
        # squares.
        if from_point == to_point:
            continue

        tail, tail_num = starts.pop(to_point, (None, None)) # Pop an element not present from the dictionary, provided a default value
        head, head_num = ends.pop(from_point, (None, None))

        if tail is not None and head is not None:
            # We need to connect these two contours.
            if tail is head:
                # We need to closed a contour: add the end point
                head.append(to_point)
            else:  # tail is not head
                # We need to join two distinct contours.
                # We want to keep the first contour segment created, so that
                # the final contours are ordered left->right, top->bottom.
                if tail_num > head_num:
                    # tail was created second. Append tail to head.
                    head.extend(tail)
                    # Remove tail from the detected contours
                    contours.pop(tail_num, None)
                    # Update starts and ends
                    starts[head[0]] = (head, head_num)
                    ends[head[-1]] = (head, head_num)
                else:  # tail_num <= head_num
                    # head was created second. Prepend head to tail.
                    tail.extendleft(reversed(head))
                    # Remove head from the detected contours
                    starts.pop(head[0], None)  # head[0] can be == to_point!
                    contours.pop(head_num, None)
                    # Update starts and ends
                    starts[tail[0]] = (tail, tail_num)
                    ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            # We need to add a new contour
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:  # tail is not None
            # tail first element is to_point: the new segment should be
            # prepended.
            tail.appendleft(from_point)
            # Update starts
            starts[from_point] = (tail, tail_num)
        else:  # tail is None and head is not None:
            # head last element is from_point: the new segment should be
            # appended
            head.append(to_point)
            # Update ends
            ends[to_point] = (head, head_num)

    return [np.array(contour) for _, contour in sorted(contours.items())]

# Set Benchmark parameters
times = 1000
inputReal = True

if (inputReal):
    # Get a Real Input
    t = np.load('./heatmaps_00000001_00000001.npy').astype(np.float64)
    image = t[...,0]
    
else:
    # Construct artificial test data
    x, y = np.ogrid[-np.pi:np.pi:95j, -np.pi:np.pi:511j]
    #x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
    
    image = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
    
    # test for case 2
    image[0,0]=1
    image[1,1]=1
    
    
    # image = np.zeros_like(image)
    # image[1,1]=1
    # image[2,2]=1
    # image[3,3]=0
print(image.shape,image.dtype)
    

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

contours, elapsed_time_my = bench_marching_squares_gpu(image, times)

print("[" + u'\u2713' + "]\n")

# Take  _assemble_contours time separately
st = time.time()

for a in range(times):
    _ = _assemble_contours(contours)

et = time.time()
elapsed_time_assemble_contours = (et - st)/times

# Print times, difference and speedup
print('RESULTS:')
print('MS execution time lib:   ', elapsed_time_lib, 'seconds   WARNING: full time with the ending data recostruction!')
print('MS execution time my:    ', elapsed_time_my, 'seconds')
print('(Estimate assemble_contours execution time:  ', elapsed_time_assemble_contours, 'seconds)')
print('Difference (all):        ', elapsed_time_lib - elapsed_time_my, 'seconds')
print('Speedup (all):           ', elapsed_time_lib / elapsed_time_my , 'seconds')
print('Speedup (excluded a_c):  ', (elapsed_time_lib - elapsed_time_assemble_contours) / elapsed_time_my , 'seconds\n')



