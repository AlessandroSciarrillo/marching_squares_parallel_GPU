from cuda import cuda, nvrtc
import numpy as np

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
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

    # Compile program
    opts = [b"--fmad=false", b"--gpu-architecture=compute_60"] #compute_75
    err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)

    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)

    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)

    # Create context
    err, context = cuda.cuCtxCreate(0, cuDevice)


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
    #print("n:   ",n)
    width = np.array(image.shape[1], dtype=np.uint32)
    height = np.array(image.shape[0], dtype=np.uint32)
    lev_np = np.array([level], dtype=np.float32)
    bufferSize = n * lev_np.itemsize

    # image è 95x511 con 48545 elementi 
    image = image.ravel()
    result_x = np.zeros(n * 2).astype(dtype=np.float32)
    result_y = np.zeros(n * 2).astype(dtype=np.float32)
    print("image:   \n",image)
    #print("result_x:  ",result_x)

    err, dImageclass = cuda.cuMemAlloc(bufferSize)
    err, dResultXclass = cuda.cuMemAlloc(bufferSize)
    err, dResultYclass = cuda.cuMemAlloc(bufferSize)

    err, stream = cuda.cuStreamCreate(0)

    err, = cuda.cuMemcpyHtoDAsync(
        dImageclass, image.ctypes.data, bufferSize, stream
    )

    dImage = np.array([int(dImageclass)], dtype=np.uint64)
    dResult_x = np.array([int(dResultXclass)], dtype=np.uint64)
    dResult_y = np.array([int(dResultYclass)], dtype=np.uint64)

    args = [dImage, dResult_x, dResult_y, n, width, height, lev_np]
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
    #ASSERT_DRV(err)

    err, = cuda.cuMemcpyDtoHAsync(
        result_x.ctypes.data, dResultXclass, bufferSize, stream
    )
    err, = cuda.cuMemcpyDtoHAsync(
        result_y.ctypes.data, dResultYclass, bufferSize, stream
    )
    err, = cuda.cuStreamSynchronize(stream)
    #print("result_x     : \n",result_x)

    # Assert values are same after running kernel   
    #hZ = a * hX + hY   
    #if not np.allclose(hOut, hZ):                   
    #    raise ValueError("Error outside tolerance for host-device vectors")

    err, = cuda.cuStreamDestroy(stream)
    err, = cuda.cuMemFree(dImageclass)
    err, = cuda.cuMemFree(dResultXclass)
    err, = cuda.cuMemFree(dResultYclass)
    err, = cuda.cuModuleUnload(module)
    err, = cuda.cuCtxDestroy(context)

    # join result_x and result_y
    segments = []
    for x, y in zip(result_x, result_y): # !!! non considera il salto alternato di n posizioni
        a = (x,y)
        b = (x,y) # !!! ci andrebbero quelli in posizione +n
        # Per dopo: Qui servirebbe filtraggio dei -1 dei valori nulli 
        segments.append( (a,b) )

    #print(segments)

    #return result_x.reshape(25,-1) #.reshape(95,511)
    return segments