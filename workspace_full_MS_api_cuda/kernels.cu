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
void prescan(int *output, int *input, size_t n, int *sums) { // n = elements_per_block
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
extern "C" __global__ 
void prescan_small(int *output, int *input, int n, int powerOfTwo) {
	extern __shared__ int temp[1024];  //TODO make dynamic from kernel launch call // allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < n) {
		temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
		temp[2 * threadID + 1] = input[2 * threadID + 1];
	}
	else {
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[2 * threadID] = temp[2 * threadID]; // write results to device memory
		output[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}

extern "C" __global__ 
void add(int *output, int length, int *n) {
    length = 64;
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

extern "C" __global__
void marching_squares(double *image, double *result_1x, double *result_1y, double *result_2x, double *result_2y, double level, size_t n, size_t width, size_t height, int *positions)
{        
    size_t square_case;
    size_t r0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c0 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t r1 = r0 + 1;
    size_t c1 = c0 + 1;

    struct tuple {
        size_t x;
        size_t y;
    } top, bottom, left, right;
    
    // height e width -1: altrimenti r1 e c1 escono dal dominio
    //  (r0,c0) _ (r0,c1)
    //     |         |
    //  (r1,c0) _ (r1,c1)


    // width: 511 => (0-510)
    //                 need : 510 (0 - max val: 509)
    if( r0 < height-1 && c0 < width-1 ){ 

        // skip mask
        
        //               need : 511
        double ul = image[ r0 * width + c0 ]; 
        double ur = image[ r0 * width + c1 ];
        double ll = image[ r1 * width + c0 ];
        double lr = image[ r1 * width + c1 ];

        // for dom_res need: [r0 * width + c0], width = 510
        width = width - 1;

        // skip control for NaN values

        square_case = 0;
        if (ul > level) square_case += 1;
        if (ur > level) square_case += 2;
        if (ll > level) square_case += 4;
        if (lr > level) square_case += 8; 

        // determinate the position of the result array where to write the number of values every thread needs
        size_t g_pos = positions[r0 * width + c0];


        if (square_case != 0 && square_case != 15){
            // case 0 and 15 have no values to write

            top.x = r0; 
            top.y = c0 + get_fraction(ul,ur,level);
            bottom.x = r1;
            bottom.y = c0 + get_fraction(ll, lr, level);
            left.x = r0 + get_fraction(ul, ll, level);
            left.y = c0;
            right.x = r0 + get_fraction(ur, lr, level);
            right.y = c1;

            //result[r0*width+c0] = square_case;

            if (square_case == 1){
                result_1x[ g_pos ] = top.x;
                result_1y[ g_pos ] = top.y;
                result_2x[ g_pos ] = left.x;
                result_2y[ g_pos ] = left.y; 
            }
            else if (square_case == 2){
                result_1x[ g_pos ] = right.x;
                result_1y[ g_pos] = right.y;
                result_2x[ g_pos ] = top.x;
                result_2y[ g_pos ] = top.y; 
            }
            else if (square_case == 3){
                result_1x[ g_pos ] = right.x;
                result_1y[ g_pos ] = right.y;
                result_2x[ g_pos ] = left.x;
                result_2y[ g_pos ] = left.y; 
            }
            else if (square_case == 4){
                result_1x[ g_pos ] = left.x;
                result_1y[ g_pos ] = left.y;
                result_2x[ g_pos ] = bottom.x;
                result_2y[ g_pos ] = bottom.y; 
            }
            else if (square_case == 5){
                result_1x[ g_pos ] = top.x;
                result_1y[ g_pos ] = top.y;
                result_2x[ g_pos ] = bottom.x;
                result_2y[ g_pos ] = bottom.y; 
            }
            else if (square_case == 6){
                // 2 couple of points to write
                result_1x[ g_pos ] = left.x;
                result_1y[ g_pos ] = left.y;
                result_2x[ g_pos ] = top.x;
                result_2y[ g_pos ] = top.y; 

                result_1x[ g_pos + 1 ] = right.x;
                result_1y[ g_pos + 1 ] = right.y;
                result_2x[ g_pos + 1 ] = bottom.x;
                result_2y[ g_pos + 1 ] = bottom.y; 
            }
            else if (square_case == 7){
                result_1x[ g_pos ] = right.x;
                result_1y[ g_pos ] = right.y;
                result_2x[ g_pos ] = bottom.x;
                result_2y[ g_pos ] = bottom.y; 
            }
            else if (square_case == 8){
                result_1x[ g_pos ] = bottom.x;
                result_1y[ g_pos ] = bottom.y;
                result_2x[ g_pos ] = right.x;
                result_2y[ g_pos ] = right.y; 
            }
            else if (square_case == 9){
                // 2 couple of points to write
                result_1x[ g_pos ] = top.x;
                result_1y[ g_pos ] = top.y;
                result_2x[ g_pos ] = right.x;
                result_2y[ g_pos ] = right.y; 

                result_1x[ g_pos + 1 ] = bottom.x;
                result_1y[ g_pos + 1 ] = bottom.y;
                result_2x[ g_pos + 1 ] = left.x;
                result_2y[ g_pos + 1 ] = left.y; 
            }
            else if (square_case == 10){
                result_1x[ g_pos ] = bottom.x;
                result_1y[ g_pos ] = bottom.y;
                result_2x[ g_pos ] = top.x;
                result_2y[ g_pos ] = top.y; 
            }
            else if (square_case == 11){
                result_1x[ g_pos ] = bottom.x;
                result_1y[ g_pos ] = bottom.y;
                result_2x[ g_pos ] = left.x;
                result_2y[ g_pos ] = left.y; 
            }
            else if (square_case == 12){
                result_1x[ g_pos ] = left.x;
                result_1y[ g_pos ] = left.y;
                result_2x[ g_pos ] = right.x;
                result_2y[ g_pos ] = right.y; 
            }
            else if (square_case == 13){
                result_1x[ g_pos ] = top.x;
                result_1y[ g_pos ] = top.y;
                result_2x[ g_pos ] = right.x;
                result_2y[ g_pos ] = right.y; 
            }
            else if (square_case == 14){
                result_1x[ g_pos ] = left.x;
                result_1y[ g_pos ] = left.y;
                result_2x[ g_pos ] = top.x;
                result_2y[ g_pos ] = top.y; 
            }              
        }
    } 
}
