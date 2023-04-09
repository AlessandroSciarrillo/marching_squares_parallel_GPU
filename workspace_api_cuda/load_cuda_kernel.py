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


def load_kernel(image_size, image_shape_1, image_shape_0, level):
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

    BLKDIM = 32   
    NUM_THREADS_x = BLKDIM  # Threads per block  x
    NUM_THREADS_y = BLKDIM  # Threads per block  y
    NUM_BLOCKS_x = (image_shape_1 + BLKDIM-1) / BLKDIM   # Blocks per grid  x
    NUM_BLOCKS_y = (image_shape_0 + BLKDIM-1) / BLKDIM   # Blocks per grid  y

    n = np.array(image_size, dtype=np.uint32) 
    width = np.array(image_shape_1, dtype=np.uint32)
    height = np.array(image_shape_0, dtype=np.uint32)
    lev_np = np.array([level], dtype=np.float64)
    bufferSize = n * lev_np.itemsize

    # image è 95x511 con 48545 elementi 
    #image = image.ravel()
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

    # err, = cuda.cuMemcpyHtoDAsync(
    #     dImageclass, image.ctypes.data, bufferSize, stream
    # )

    dImage = np.array([int(dImageclass)], dtype=np.uint64)
    dResult_1x = np.array([int(dResult1Xclass)], dtype=np.uint64)
    dResult_1y = np.array([int(dResult1Yclass)], dtype=np.uint64)
    dResult_2x = np.array([int(dResult2Xclass)], dtype=np.uint64)
    dResult_2y = np.array([int(dResult2Yclass)], dtype=np.uint64)

    args = [dImage, dResult_1x, dResult_1y, dResult_2x, dResult_2y, lev_np, n, width, height]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    return (kernel, bufferSize, stream, args, result_1x, result_1y, result_2x, result_2y, dResult1Xclass, dResult1Yclass, dResult2Xclass, dResult2Yclass, dImageclass, NUM_BLOCKS_x, NUM_BLOCKS_y, NUM_THREADS_x, NUM_THREADS_y)

    


