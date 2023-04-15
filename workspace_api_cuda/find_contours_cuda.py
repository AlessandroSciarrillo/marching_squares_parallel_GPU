from cuda import cuda, nvrtc
import numpy as np

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

def _get_contour_segments(image,level):
    #strategia:
    #     allocare due coppie di np.float64_t
    #     nel caso classico solo una coppia viene occupata
    #     nel caso 6 e 9 entrambe vengono occupate
    #     nel caso 0 e 15 entrambe non occupate
    #     nelle non-occupate mettere valore es: -1
    with open('kernel.cu', 'r') as file:
        saxpy = file.read()

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], [])
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
    err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
    ASSERT_DRV(err)

    print(image.shape)
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
    print("image:   \n",image)

    err, dImageclass = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)
    err, dResult1Xclass = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)
    err, dResult1Yclass = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)
    err, dResult2Xclass = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)
    err, dResult2Yclass = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    err, = cuda.cuMemcpyHtoDAsync(
        dImageclass, image.ctypes.data, bufferSize, stream
    )
    ASSERT_DRV(err)

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

    # Assert values are same after running kernel   
    #hZ = a * hX + hY   
    #if not np.allclose(hOut, hZ):                   
    #    raise ValueError("Error outside tolerance for host-device vectors")

    err, = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(dImageclass)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(dResult1Xclass)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(dResult1Yclass)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(dResult2Xclass)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(dResult2Yclass)
    ASSERT_DRV(err)
    err, = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    err, = cuda.cuCtxDestroy(context)
    ASSERT_DRV(err)

    segments = []
    for (x1, y1, x2, y2) in zip(result_1x, result_1y, result_2x, result_2y):  
        if x1 > 0.0 and y1 > 0.0 and x2 > 0.0 and y2 > 0.0 :
            point1 = (x1,y1)
            point2 = (x2,y2) 
            segments.append( (point1,point2) )

    #print(segments)
    return segments