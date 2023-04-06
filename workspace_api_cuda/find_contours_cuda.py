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
    saxpy = """\
    extern "C" __global__
    void saxpy(float *image, float *result, size_t n)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            result[tid] = image[tid] + 2.0;
        }
    }
    """


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

    BLKDIM = 512   
    NUM_THREADS = BLKDIM  # Threads per block  //512
    NUM_BLOCKS = (image.size + BLKDIM-1) / BLKDIM   # Blocks per grid  /32768

    n = np.array(image.size, dtype=np.uint32) 
    print("n:   ",n)
    lev_np = np.array([level], dtype=np.float32)
    bufferSize = n * lev_np.itemsize

    # image è 95x511 con 48545 elementi 
    image = image.ravel()
    result = np.zeros(n).astype(dtype=np.float32)
    print("image:   \n",image)
    print("result:  ",result)

    err, dImageclass = cuda.cuMemAlloc(bufferSize)
    err, dResultclass = cuda.cuMemAlloc(bufferSize)

    err, stream = cuda.cuStreamCreate(0)

    err, = cuda.cuMemcpyHtoDAsync(
        dImageclass, image.ctypes.data, bufferSize, stream
    )

    dImage = np.array([int(dImageclass)], dtype=np.uint64)
    dResult = np.array([int(dResultclass)], dtype=np.uint64)

    args = [dImage, dResult, n]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    err, = cuda.cuLaunchKernel(
        kernel,
        NUM_BLOCKS,  # grid x dim
        1,  # grid y dim
        1,  # grid z dim
        NUM_THREADS,  # block x dim
        1,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )

    err, = cuda.cuMemcpyDtoHAsync(
        result.ctypes.data, dResultclass, bufferSize, stream
    )
    err, = cuda.cuStreamSynchronize(stream)
    print("result     : \n",result)

    # Assert values are same after running kernel   
    #hZ = a * hX + hY   
    #if not np.allclose(hOut, hZ):                   
    #    raise ValueError("Error outside tolerance for host-device vectors")

    err, = cuda.cuStreamDestroy(stream)
    err, = cuda.cuMemFree(dImageclass)
    err, = cuda.cuMemFree(dResultclass)
    err, = cuda.cuModuleUnload(module)
    err, = cuda.cuCtxDestroy(context)

    return result.reshape(95,511)