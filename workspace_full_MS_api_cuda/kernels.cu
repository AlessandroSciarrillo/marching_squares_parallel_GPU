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