
import time
import numpy as np
import matplotlib.pyplot as plt #WARNING: se viene rimosso Cuda genera Illegal Access Memory
from cuda import cuda, nvrtc, cudart # cudart only for GPUs info

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

def nextPowerOfTwo(x):
	power = 1;
	while (power < x):
		power *= 2;
	return int(power);



# Set Benchmark parameters
times = 10000
inputReal = False

if (inputReal):
    # Get a Real Input
    t = np.load('./heatmaps_00000001_00000001.npy')
    r = t[:,:,10]
    image = r # for kernel names
else:
    # Construct artificial test data
    x, y = np.ogrid[-np.pi:np.pi:95j, -np.pi:np.pi:511j]
    r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
    image = r

# test for case 2
image[0,0]=1
image[1,1]=1

N = r.size
h = r.shape[0]
w = r.shape[1]
level = 0.5
print("N: ",N," h: ", h, "w: ", w)

# Load Cuda kernel
# kernel, bufferSize, stream, args, \
# result_1x, result_1y, result_2x, result_2y, \
# dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, \
# dImageclass, \
# NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, \
# module, context = load_kernel(r.size, r.shape[1], r.shape[0], 0.5)

#st = time.time()
# Only kernel execution
# segments, _ = launch_kernel( kernel, bufferSize, stream, args,
#                                 result_1x, result_1y, result_2x, result_2y, 
#                                 dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, dImageclass, 
#                                 NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, 
#                                 r.astype(np.float64), 0.5)
# et = time.time()
# elapsed_time_my = (et - st)/times

#################### Full Marching Squares #########à

# st = time.time()

# -1 preparazione array lungo n
#     v0  | | | | |        
# -2 lancio kernel dove ogni cuda core scrive nella sua posizione quanto spazio ha bisogno
#     v0  |2|0|2|1| 
# -3 lancio kernel che calcoli la somma dell'array con Reduce
#     redude_res  [5]
# -4 lancio kernel che faccia la Exclusive scan sull'array con la quantità di spazio occupato (GRANDEZZA > 2048)
#     v1 |0|2|2|4|
# -5 lancio kernel che faccia la Exclusive scan sull'array con la quantità di spazio occupato (GRANDEZZA < 2048)
#     v1 |0|2|2|4|
# -6 lancio kernel che scriva i propri valori su un nuovo array res di lunghezza redude_res partendo dalla posizione scritta su v1
#     res |t0|t0|t2|t2|t3|
#
# NOTE: unire kernel step 2 e 3

with open('kernels.cu', 'r') as file:
    saxpy = file.read()

# Create program
err, prog = nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"kernels.cu", 0, [], [])
ASSERT_DRV(err)
# Compile program
opts = [b"--fmad=false", b"--gpu-architecture=compute_60"] #compute_75
err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)
ASSERT_DRV(err)
# Get PTX from compilation
err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
ASSERT_DRV(err)
ptx = b" " * ptxSize
err, = nvrtc.nvrtcGetPTX(prog, ptx)
ASSERT_DRV(err)


# Initialize CUDA Driver API
err, = cuda.cuInit(0)
ASSERT_DRV(err)
# Retrieve handle for device 0
err, cuDevice = cuda.cuDeviceGet(0)
ASSERT_DRV(err)
# Create context
err, context = cuda.cuCtxCreate(0, cuDevice)
ASSERT_DRV(err)
# Load PTX as module data and retrieve function
ptx = np.char.array(ptx)
# Note: Incompatible --gpu-architecture would be detected here
err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
ASSERT_DRV(err)
# kernel 1
err, kernel_1 = cuda.cuModuleGetFunction(module, b"required_memory")
ASSERT_DRV(err)
# kernel 2
err, kernel_2 = cuda.cuModuleGetFunction(module, b"reduce")
ASSERT_DRV(err)
# kernel 3
err, kernel_3 = cuda.cuModuleGetFunction(module, b"prescan")
ASSERT_DRV(err)
# kernel 4
err, kernel_4 = cuda.cuModuleGetFunction(module, b"prescan_small")
ASSERT_DRV(err)
# kernel 5
err, kernel_5 = cuda.cuModuleGetFunction(module, b"add")
ASSERT_DRV(err)
# kernel 6
err, kernel_6 = cuda.cuModuleGetFunction(module, b"marching_squares")
ASSERT_DRV(err)

# BLKDIM = 32   
# NUM_THREADS_x = BLKDIM                  # Threads per block  x
# NUM_THREADS_y = BLKDIM                  # Threads per block  y
# NUM_BLOCKS_x = (w + BLKDIM-1) / BLKDIM  # Blocks per grid  x
# NUM_BLOCKS_y = (h + BLKDIM-1) / BLKDIM  # Blocks per grid  y

BLKDIM = 32  
REDUCE_BLOCKS = (N + BLKDIM-1) / BLKDIM
NUM_BLOCKS_x_kernel_3 = int( (N - (N%64)) / 64 )   
POWEROFTWO_kernel_4 = nextPowerOfTwo(NUM_BLOCKS_x_kernel_3)

print("Number of last elements out of exclusive scan process: ",N%64)
print(POWEROFTWO_kernel_4)

# image è 95x511 con 48545 elementi 
n = np.array( N , dtype=np.uint32) 
reduce_blocks = np.array( REDUCE_BLOCKS , dtype=np.uint32) 
powerOfTwo = np.array(POWEROFTWO_kernel_4, dtype=np.uint32)
width = np.array( w , dtype=np.uint32)
height = np.array( h , dtype=np.uint32)
lev_np = np.array([level], dtype=np.float64)

bufferSize_image = n * lev_np.itemsize                  # n * sizeof( np.float64)
# kernel 1
bufferSize_result_required_memory = n * n.itemsize      # n * sizeof( np.uint32)
# kernel 2
bufferSize_result_reduce = reduce_blocks * n.itemsize   # REDUCE_BLOCKS * sizeof( np.uint32)
# kernel 3
bufferSize_result_exc_scan = n * n.itemsize             # n * sizeof( np.uint32)
bufferSize_aux_exc_scan = NUM_BLOCKS_x_kernel_3 * n.itemsize             # NUM_BLOCKS_x_kernel_3 * sizeof( np.uint32)
bufferSize_incr_exc_scan = NUM_BLOCKS_x_kernel_3 * n.itemsize 
# kernel 6
# bufferSize_resultsMS = n * lev_np.itemsize # TODO importante farlo solo delle dimanensioni del risultato della reduce

result_required_memory = np.zeros(n).astype(dtype=np.uint32)
result_reduce = np.zeros(reduce_blocks).astype(dtype=np.uint32)
result_exc_scan =  np.zeros(n).astype(dtype=np.uint32)
aux_exc_scan =  np.zeros(NUM_BLOCKS_x_kernel_3).astype(dtype=np.uint32)
incr_exc_scan =  np.zeros(NUM_BLOCKS_x_kernel_3).astype(dtype=np.uint32)
# result_1x = np.zeros(n).astype(dtype=np.float64)
# result_1y = np.zeros(n).astype(dtype=np.float64)
# result_2x = np.zeros(n).astype(dtype=np.float64)
# result_2y = np.zeros(n).astype(dtype=np.float64)


err, dImageclass = cuda.cuMemAlloc(bufferSize_image)
ASSERT_DRV(err)
# kernel 1
err, dResult_required_memorys_class = cuda.cuMemAlloc(bufferSize_result_required_memory)
ASSERT_DRV(err)
# kenel 2
err, dResult_reduce_class = cuda.cuMemAlloc(bufferSize_result_reduce)
ASSERT_DRV(err)
# kenel 3
err, dResult_exc_scan_class = cuda.cuMemAlloc(bufferSize_result_exc_scan)
ASSERT_DRV(err)
err, dAux_exc_scan_class = cuda.cuMemAlloc(bufferSize_aux_exc_scan)
ASSERT_DRV(err)
# kenel 4
err, dIncr_exc_scan_class = cuda.cuMemAlloc(bufferSize_incr_exc_scan)
ASSERT_DRV(err)
# kernel 6
# err, dResult1Xclass = cuda.cuMemAlloc(bufferSize_resultsMS)
# ASSERT_DRV(err)
# err, dResult1Yclass = cuda.cuMemAlloc(bufferSize_resultsMS)
# ASSERT_DRV(err)
# err, dResult2Xclass = cuda.cuMemAlloc(bufferSize_resultsMS)
# ASSERT_DRV(err)
# err, dResult2Yclass = cuda.cuMemAlloc(bufferSize_resultsMS)
# ASSERT_DRV(err)

err, stream = cuda.cuStreamCreate(0)

dImage = np.array([int(dImageclass)], dtype=np.uint64)
# kernel 1
dResult_required_memorys = np.array([int(dResult_required_memorys_class)], dtype=np.uint64)
# kernel 2
dResult_reduce = np.array([int(dResult_reduce_class)], dtype=np.uint64)
# kernel 3
dResult_exc_scan = np.array([int(dResult_exc_scan_class)], dtype=np.uint64)
dAux_exc_scan = np.array([int(dAux_exc_scan_class)], dtype=np.uint64)
# kenel 4
dIncr_exc_scan = np.array([int(dIncr_exc_scan_class)], dtype=np.uint64)
# kernel 6
# dResult_1x = np.array([int(dResult1Xclass)], dtype=np.uint64)
# dResult_1y = np.array([int(dResult1Yclass)], dtype=np.uint64)
# dResult_2x = np.array([int(dResult2Xclass)], dtype=np.uint64)
# dResult_2y = np.array([int(dResult2Yclass)], dtype=np.uint64)

# kernel 1
args_1 = [dImage, dResult_required_memorys, lev_np, n, width, height]
args_1 = np.array([arg.ctypes.data for arg in args_1], dtype=np.uint64)
# kernel 2
args_2 = [dResult_required_memorys, dResult_reduce, n]
args_2 = np.array([arg.ctypes.data for arg in args_2], dtype=np.uint64)
# kernel 3
args_3 = [dResult_exc_scan, dResult_required_memorys, n, dAux_exc_scan]
args_3 = np.array([arg.ctypes.data for arg in args_3], dtype=np.uint64)
# kernel 4
args_4 = [dIncr_exc_scan, dAux_exc_scan, n, powerOfTwo]
args_4 = np.array([arg.ctypes.data for arg in args_4], dtype=np.uint64) 
# kernel 5
args_5 = [dResult_exc_scan, n, dIncr_exc_scan]
args_5 = np.array([arg.ctypes.data for arg in args_5], dtype=np.uint64) 
# kernel 6
# args_6 = [dImage, dResult_1x, dResult_1y, dResult_2x, dResult_2y, lev_np, n, width, height, dResult_exc_scan]
# args_6 = np.array([arg.ctypes.data for arg in args_6], dtype=np.uint64)

image = image.ravel()

# start taking time
st = time.time()

for i_time in range(times):

    err, = cuda.cuMemcpyHtoDAsync(
        dImageclass, image.ctypes.data, bufferSize_image, stream
    )
    ASSERT_DRV(err)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    NUM_THREADS_x = BLKDIM                  # Threads per block  x
    NUM_THREADS_y = BLKDIM                  # Threads per block  y
    NUM_BLOCKS_x = (w + BLKDIM-1) / BLKDIM  # Blocks per grid  x
    NUM_BLOCKS_y = (h + BLKDIM-1) / BLKDIM  # Blocks per grid  y

    # kernel 1
    err, = cuda.cuLaunchKernel(
        kernel_1,
        NUM_BLOCKS_x,  # grid x dim
        NUM_BLOCKS_y,  # grid y dim
        1,  # grid z dim
        NUM_THREADS_x,  # block x dim
        NUM_THREADS_y,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args_1.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )
    ASSERT_DRV(err)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)
    
    NUM_THREADS_x = BLKDIM                  # Threads per block  x
    NUM_THREADS_y = 1                       # Threads per block  y
    NUM_BLOCKS_x = REDUCE_BLOCKS            # Blocks per grid  x        (N + BLKDIM-1) / BLKDIM           
    NUM_BLOCKS_y = 1                        # Blocks per grid  y

    # kernel 2
    err, = cuda.cuLaunchKernel(
        kernel_2,
        NUM_BLOCKS_x,  # grid x dim
        NUM_BLOCKS_y,  # grid y dim
        1,  # grid z dim
        NUM_THREADS_x,  # block x dim
        NUM_THREADS_y,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args_2.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )
    ASSERT_DRV(err)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    NUM_THREADS_x = BLKDIM   # ATTENZIONE                 # Threads per block  x
    NUM_THREADS_y = 1                       # Threads per block  y
    NUM_BLOCKS_x = NUM_BLOCKS_x_kernel_3          # Blocks per grid  x                 /*+ BLKDIM-1*/
    NUM_BLOCKS_y = 1                        # Blocks per grid  y

    # kernel 3
    err, = cuda.cuLaunchKernel(
        kernel_3,
        NUM_BLOCKS_x,  # grid x dim
        NUM_BLOCKS_y,  # grid y dim
        1,  # grid z dim
        NUM_THREADS_x,  # block x dim
        NUM_THREADS_y,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args_3.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )
    ASSERT_DRV(err)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    NUM_THREADS_x = (NUM_BLOCKS_x_kernel_3 + 1) // 2               # Threads per block  x
    NUM_THREADS_y = 1                       # Threads per block  y
    NUM_BLOCKS_x = 1         # Blocks per grid  x                 /*+ BLKDIM-1*/
    NUM_BLOCKS_y = 1                        # Blocks per grid  y

    # kernel 4
    err, = cuda.cuLaunchKernel(
        kernel_4,
        NUM_BLOCKS_x,  # grid x dim
        NUM_BLOCKS_y,  # grid y dim
        1,  # grid z dim
        NUM_THREADS_x,  # block x dim
        NUM_THREADS_y,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args_4.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )
    ASSERT_DRV(err)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    NUM_THREADS_x = 64             # Threads per block  x
    NUM_THREADS_y = 1                       # Threads per block  y
    NUM_BLOCKS_x = NUM_BLOCKS_x_kernel_3         # Blocks per grid  x                 /*+ BLKDIM-1*/
    NUM_BLOCKS_y = 1                        # Blocks per grid  y

    # kernel 5
    err, = cuda.cuLaunchKernel(
        kernel_5,
        NUM_BLOCKS_x,  # grid x dim
        NUM_BLOCKS_y,  # grid y dim
        1,  # grid z dim
        NUM_THREADS_x,  # block x dim
        NUM_THREADS_y,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args_5.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )
    ASSERT_DRV(err)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    # use the result of the reduce for allocate the correct amount of memory
    err, = cuda.cuMemcpyDtoHAsync(
        result_reduce.ctypes.data, dResult_reduce_class, bufferSize_result_reduce, stream
    )
    ASSERT_DRV(err)
    np_result_reduce = np.array(result_reduce) 
    N_RES = np_result_reduce.sum()   #TODO fare questa somma su GPU sarebbe meglio

    bufferSize_resultsMS = N_RES * lev_np.itemsize 

    result_1x = np.zeros(N_RES).astype(dtype=np.float64)
    result_1y = np.zeros(N_RES).astype(dtype=np.float64)
    result_2x = np.zeros(N_RES).astype(dtype=np.float64)
    result_2y = np.zeros(N_RES).astype(dtype=np.float64)

    err, dResult1Xclass = cuda.cuMemAlloc(bufferSize_resultsMS)
    ASSERT_DRV(err)
    err, dResult1Yclass = cuda.cuMemAlloc(bufferSize_resultsMS)
    ASSERT_DRV(err)
    err, dResult2Xclass = cuda.cuMemAlloc(bufferSize_resultsMS)
    ASSERT_DRV(err)
    err, dResult2Yclass = cuda.cuMemAlloc(bufferSize_resultsMS)
    ASSERT_DRV(err)

    dResult_1x = np.array([int(dResult1Xclass)], dtype=np.uint64)
    dResult_1y = np.array([int(dResult1Yclass)], dtype=np.uint64)
    dResult_2x = np.array([int(dResult2Xclass)], dtype=np.uint64)
    dResult_2y = np.array([int(dResult2Yclass)], dtype=np.uint64)

    args_6 = [dImage, dResult_1x, dResult_1y, dResult_2x, dResult_2y, lev_np, n, width, height, dResult_exc_scan]
    args_6 = np.array([arg.ctypes.data for arg in args_6], dtype=np.uint64)

    image_shape_1 = w
    image_shape_0 = h

    BLKDIM = 32   
    NUM_THREADS_x = BLKDIM                              # Threads per block  x
    NUM_THREADS_y = BLKDIM                              # Threads per block  y
    NUM_BLOCKS_x = (image_shape_1 + BLKDIM-1) / BLKDIM  # Blocks per grid  x
    NUM_BLOCKS_y = (image_shape_0 + BLKDIM-1) / BLKDIM  # Blocks per grid  y

    err, = cuda.cuLaunchKernel(
            kernel_6,
            NUM_BLOCKS_x,  # grid x dim
            NUM_BLOCKS_y,  # grid y dim
            1,  # grid z dim
            NUM_THREADS_x,  # block x dim
            NUM_THREADS_y,  # block y dim
            1,  # block z dim
            0,  # dynamic shared memory
            stream,  # stream
            args_6.ctypes.data,  # kernel arguments
            0,  # extra (ignore)
    )
    ASSERT_DRV(err)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    #TODO non serve riportare giù i risultati dei kernel precedenti all ultimo, fatto solo per check risultato, da commentare per misurazione tempi

    # kernel 1
    # err, = cuda.cuMemcpyDtoHAsync( 
    #     result_required_memory.ctypes.data, dResult_required_memorys_class, bufferSize_result_required_memory, stream
    # )
    # ASSERT_DRV(err)
    # # kernel 2
    # err, = cuda.cuMemcpyDtoHAsync(
    #     result_reduce.ctypes.data, dResult_reduce_class, bufferSize_result_reduce, stream
    # )
    # ASSERT_DRV(err)

    # # Calc final result reduce
    # np_result_reduce = np.array(result_reduce) 
    # np_result_reduce = np_result_reduce.sum()   #TODO fare questa somma su GPU sarebbe meglio

    # # kernel 3
    # err, = cuda.cuMemcpyDtoHAsync(
    #     result_exc_scan.ctypes.data, dResult_exc_scan_class, bufferSize_result_exc_scan, stream
    # )
    # ASSERT_DRV(err)

    # err, = cuda.cuMemcpyDtoHAsync(
    #     aux_exc_scan.ctypes.data, dAux_exc_scan_class, bufferSize_aux_exc_scan, stream
    # )
    # ASSERT_DRV(err)

    # # kernel 4
    # err, = cuda.cuMemcpyDtoHAsync(
    #     incr_exc_scan.ctypes.data, dIncr_exc_scan_class, bufferSize_incr_exc_scan, stream
    # )
    # ASSERT_DRV(err)

    # kernel 6
    err, = cuda.cuMemcpyDtoHAsync(
        result_1x.ctypes.data, dResult1Xclass, bufferSize_resultsMS, stream
    )
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyDtoHAsync(
        result_1y.ctypes.data, dResult1Yclass, bufferSize_resultsMS, stream
    )
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyDtoHAsync(
        result_2x.ctypes.data, dResult2Xclass, bufferSize_resultsMS, stream
    )
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyDtoHAsync(
        result_2y.ctypes.data, dResult2Yclass, bufferSize_resultsMS, stream
    )
    ASSERT_DRV(err)


    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)


et = time.time()
elapsed_time_my = (et - st)/times

# check results
# with open("res1.txt", "w") as txt_file:
#     for val in result_required_memory:
#         txt_file.write("{} \n".format(val))
# with open("res2.txt", "w") as txt_file:
#     txt_file.write("Result reduce: {} \n\n".format(np_result_reduce))
#     for val in result_reduce:
#         txt_file.write("{} \n".format(val))
# with open("res3.txt", "w") as txt_file:
#     i32 = 0
#     for val in result_exc_scan:
#         txt_file.write("{} ".format(val))
#         i32 = i32 +1
#         if(i32==32):
#             i32=0
#             txt_file.write("\n")
# with open("res3_sums.txt", "w") as txt_file:
#     i32 = 0
#     for val in aux_exc_scan:
#         txt_file.write("{} ".format(val))
#         i32 = i32 +1
#         if(i32==32):
#             i32=0
#             txt_file.write("\n")
# with open("res4.txt", "w") as txt_file:
    # i32 = 0
    # for val in incr_exc_scan:
    #     txt_file.write("{} ".format(val))
    #     i32 = i32 +1
    #     if(i32==32):
    #         i32=0
    #         txt_file.write("\n")
with open("resMS_1x.txt", "w") as txt_file:
    i32 = 0
    for val in result_1x:
        txt_file.write("{} ".format(val))
        i32 = i32 +1
        if(i32==32):
            i32=0
            txt_file.write("\n")
with open("resMS_1y.txt", "w") as txt_file:
    i32 = 0
    for val in result_1y:
        txt_file.write("{} ".format(val))
        i32 = i32 +1
        if(i32==32):
            i32=0
            txt_file.write("\n")
with open("resMS_2x.txt", "w") as txt_file:
    i32 = 0
    for val in result_2x:
        txt_file.write("{} ".format(val))
        i32 = i32 +1
        if(i32==32):
            i32=0
            txt_file.write("\n")
with open("resMS_2y.txt", "w") as txt_file:
    i32 = 0
    for val in result_2y:
        txt_file.write("{} ".format(val))
        i32 = i32 +1
        if(i32==32):
            i32=0
            txt_file.write("\n")

# Free Cuda Kernel memory
err, = cuda.cuStreamDestroy(stream)
err, = cuda.cuMemFree(dImageclass)
err, = cuda.cuMemFree(dResult_required_memorys_class)
err, = cuda.cuMemFree(dResult_reduce_class)
err, = cuda.cuMemFree(dResult_exc_scan_class)
err, = cuda.cuMemFree(dAux_exc_scan_class)
err, = cuda.cuMemFree(dIncr_exc_scan_class)
err, = cuda.cuMemFree(dResult1Xclass)
err, = cuda.cuMemFree(dResult1Yclass)
err, = cuda.cuMemFree(dResult2Xclass)
err, = cuda.cuMemFree(dResult2Yclass)
err, = cuda.cuModuleUnload(module) 
err, = cuda.cuCtxDestroy(context)  

print('MS execution time my:    ', elapsed_time_my, 'seconds')
