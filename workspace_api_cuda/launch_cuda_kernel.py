from cuda import cuda, nvrtc
import numpy as np

# only fot test
import time

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

def launch_kernel(  kernel, bufferSize, stream, args, 
                    result_1x, result_1y, result_2x, result_2y, 
                    dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, dImageclass, 
                    NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y, image, level):

    image = image.ravel()

    st = time.time()

    #TODO Ask Prof"For increased application performance, you can input data on the device to eliminate data transfers."
    err, = cuda.cuMemcpyHtoDAsync(
        dImageclass, image.ctypes.data, bufferSize, stream
    )
    ASSERT_DRV(err)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

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
    ASSERT_DRV(err)

    # For Illegal memory access error
    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    err, = cuda.cuMemcpyDtoHAsync(
        result_1x.ctypes.data, dResult1Xclass, bufferSize, stream
    )
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyDtoHAsync(
        result_1y.ctypes.data, dResult1Yclass, bufferSize, stream
    )
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyDtoHAsync(
        result_2x.ctypes.data, dResult2Xclass, bufferSize, stream
    )
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyDtoHAsync(
        result_2y.ctypes.data, dResult2Yclass, bufferSize, stream
    )
    ASSERT_DRV(err)
    
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    et = time.time()
    elapsed_time_kernel = (et - st)

    
    #st = time.time()

    segments = []
    # for (x1, y1, x2, y2) in zip(result_1x, result_1y, result_2x, result_2y):  
    #     if x1 > 0.0 and y1 > 0.0 and x2 > 0.0 and y2 > 0.0 :
    #         point1 = (x1,y1)
    #         point2 = (x2,y2) 
    #         segments.append( (point1,point2) )


    # stacked = np.vstack((result_1x, result_1y, result_2x, result_2y))
    # segments2 = stacked[:,np.sum(stacked<0, axis=0)==0].T
    
    #et = time.time()
    #elapsed_time_zip_res = (et - st)
    #print('Zip Risultati:', elapsed_time_zip_res, 'seconds')

    #print(segments)
    return segments, elapsed_time_kernel