from cuda import cuda, nvrtc
import numpy as np

def launch_kernel(kernel, image, level):

    BLKDIM = 32   
    NUM_THREADS_x = BLKDIM  # Threads per block  x
    NUM_THREADS_y = BLKDIM  # Threads per block  y
    NUM_BLOCKS_x = (image.shape[1] + BLKDIM-1) / BLKDIM   # Blocks per grid  x
    NUM_BLOCKS_y = (image.shape[0] + BLKDIM-1) / BLKDIM   # Blocks per grid  y

    n = np.array(image.size, dtype=np.uint32) 
    width = np.array(image.shape[1], dtype=np.uint32)
    height = np.array(image.shape[0], dtype=np.uint32)
    lev_np = np.array([level], dtype=np.float64)
    bufferSize = n * lev_np.itemsize

    # image è 95x511 con 48545 elementi 
    image = image.ravel()
    result_1x = np.zeros(n).astype(dtype=np.float64)
    result_1y = np.zeros(n).astype(dtype=np.float64)
    result_2x = np.zeros(n).astype(dtype=np.float64)
    result_2y = np.zeros(n).astype(dtype=np.float64)

    err, dImageclass = cuda.cuMemAlloc(bufferSize)
    err, dResult1Xclass = cuda.cuMemAlloc(bufferSize)
    err, dResult1Yclass = cuda.cuMemAlloc(bufferSize)
    err, dResult2Xclass = cuda.cuMemAlloc(bufferSize)
    err, dResult2Yclass = cuda.cuMemAlloc(bufferSize)

    err, stream = cuda.cuStreamCreate(0)

    err, = cuda.cuMemcpyHtoDAsync(
        dImageclass, image.ctypes.data, bufferSize, stream
    )

    dImage = np.array([int(dImageclass)], dtype=np.uint64)
    dResult_1x = np.array([int(dResult1Xclass)], dtype=np.uint64)
    dResult_1y = np.array([int(dResult1Yclass)], dtype=np.uint64)
    dResult_2x = np.array([int(dResult2Xclass)], dtype=np.uint64)
    dResult_2y = np.array([int(dResult2Yclass)], dtype=np.uint64)

    args = [dImage, dResult_1x, dResult_1y, dResult_2x, dResult_2y, lev_np, n, width, height]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    err, = cuda.cuLaunchKernel(
        kernel,
        NUM_BLOCKS_x,  # grid x dim
        NUM_BLOCKS_y,  # grid y dim
        1,  # grid z dim
        NUM_THREADS_x,  # block x dim
        NUM_THREADS_y,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )

    err, = cuda.cuMemcpyDtoHAsync(
        result_1x.ctypes.data, dResult1Xclass, bufferSize, stream
    )
    err, = cuda.cuMemcpyDtoHAsync(
        result_1y.ctypes.data, dResult1Yclass, bufferSize, stream
    )
    err, = cuda.cuMemcpyDtoHAsync(
        result_2x.ctypes.data, dResult2Xclass, bufferSize, stream
    )
    err, = cuda.cuMemcpyDtoHAsync(
        result_2y.ctypes.data, dResult2Yclass, bufferSize, stream
    )
    err, = cuda.cuStreamSynchronize(stream)

    #TODO portare fuori
    # err, = cuda.cuStreamDestroy(stream)
    # err, = cuda.cuMemFree(dImageclass)
    # err, = cuda.cuMemFree(dResult1Xclass)
    # err, = cuda.cuMemFree(dResult1Yclass)
    # err, = cuda.cuMemFree(dResult2Xclass)
    # err, = cuda.cuMemFree(dResult2Yclass)
    # err, = cuda.cuModuleUnload(module) #TODO
    # err, = cuda.cuCtxDestroy(context)  #TODO

    segments = []
    for (x1, y1, x2, y2) in zip(result_1x, result_1y, result_2x, result_2y):  
        if x1 > 0.0 and y1 > 0.0 and x2 > 0.0 and y2 > 0.0 :
            point1 = (x1,y1)
            point2 = (x2,y2) 
            segments.append( (point1,point2) )

    #print(segments)
    return segments