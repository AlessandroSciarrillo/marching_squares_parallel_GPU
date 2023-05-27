extern "C" __device__
float get_fraction(double from_value, double to_value, double level){
    if (to_value == from_value)
        return 0;
    return ((level - from_value) / (to_value - from_value));
}

extern "C" __global__
void required_memory(double *image, size_t *result_required_memory, double level, size_t n, size_t width, size_t height)
{  
    size_t square_case;
    size_t r0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c0 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t r1 = r0 + 1;
    size_t c1 = c0 + 1;
    
    if( r0 < height-1 && c0 < width-1 ){ 

        double ul = image[ r0 * width + c0 ]; 
        double ur = image[ r0 * width + c1 ];
        double ll = image[ r1 * width + c0 ];
        double lr = image[ r1 * width + c1 ];

        width = width - 1;

        square_case = 0;
        if (ul > level) square_case += 1;
        if (ur > level) square_case += 2;
        if (ll > level) square_case += 4;
        if (lr > level) square_case += 8; 

        if (square_case == 0 || square_case == 15){
            // 0
            result_required_memory[ r0 * width + c0 ] = 0;
        }
        else if (square_case == 6 || square_case == 9){
            // 2
            result_required_memory[ r0 * width + c0 ] = 2;
        }
        else {
            // 1
            result_required_memory[ r0 * width + c0 ] = 1;
        }    
    } 
}

extern "C" __global__
void reduce(size_t *required_memory, size_t *result_reduce, size_t n)
{ 
    __shared__ size_t temp[32]; // BLKDIM=32
    const size_t lindex = threadIdx.x;
    const size_t bindex = blockIdx.x;
    const size_t gindex = blockIdx.x * blockDim.x + threadIdx.x;
    size_t bsize =  blockDim.x / 2;
    temp[lindex] = required_memory[gindex];
    __syncthreads();

    while( bsize > 0 ){
        if( lindex < bsize && (lindex+bsize)<n ){
            temp[lindex] += temp[lindex+bsize];
        }
        bsize = bsize / 2;
        __syncthreads();
    }
    if(0==lindex){
        result_reduce[bindex] = temp[0];
    }
}


/*
* Exclusive Scan
* Source1 : https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
* Source2 : https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
* Source3 : https://github.com/mattdean1/cuda
*/

extern "C" __global__ 
void prescan(int *input, int *output, size_t n, int *sums) { // n = elements_per_block
    n = 64;
	size_t blockID = blockIdx.x;
	size_t threadID = threadIdx.x;
	size_t blockOffset = blockID * n;

	__shared__ size_t temp[64];


    temp[2 * threadID] = input[blockOffset + (2 * threadID)];
    temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

    size_t offset = 1;
    for (size_t d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (threadID < d)
        {
            size_t ai = offset * (2 * threadID + 1) - 1;
            size_t bi = offset * (2 * threadID + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();


    if (threadID == 0) {
        sums[blockID] = temp[n - 1]; //TODO controllare grandezza sums sia corretta
        temp[n - 1] = 0;
    }

    for (size_t d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            size_t ai = offset * (2 * threadID + 1) - 1;
            size_t bi = offset * (2 * threadID + 2) - 1;
            size_t t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    output[blockOffset + (2 * threadID)] = temp[2 * threadID];
    output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}


//TODO 
/* This part is need

https://github.com/mattdean1/cuda/blob/master/parallel-scan/Submission.cu

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	if (bcao) {
		prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}
	else {
		prescan_large_unoptimized<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}

    ======================== TODO questa parte non è
    ||
    V

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}

*/