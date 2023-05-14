#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define BLKDIM 32

__device__ float get_fraction(double from_value, double to_value, double level){
    if (to_value == from_value)
        return 0;
    return ((level - from_value) / (to_value - from_value));
}

__global__ void saxpy(double *image, double *result_1x, double *result_1y, double *result_2x, double *result_2y, double level, size_t n, size_t width, size_t height)
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

        if (square_case == 0 || square_case == 15){
            //TODO 
            result_1x[r0 * width + c0] = 0.0;
            result_1y[r0 * width + c0] = 0.0; 
            result_2x[r0 * width + c0] = 0.0;
            result_2y[r0 * width + c0] = 0.0;
        }

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
            result_1x[ r0 * width + c0 ] = top.x;
            result_1y[ r0 * width + c0 ] = top.y;
            result_2x[ r0 * width + c0 ] = left.x;
            result_2y[ r0 * width + c0 ] = left.y; 
        }
        else if (square_case == 2){
            result_1x[ r0 * width + c0 ] = right.x;
            result_1y[ r0 * width + c0 ] = right.y;
            result_2x[ r0 * width + c0 ] = top.x;
            result_2y[ r0 * width + c0 ] = top.y; 
        }
        else if (square_case == 3){
            result_1x[ r0 * width + c0 ] = right.x;
            result_1y[ r0 * width + c0 ] = right.y;
            result_2x[ r0 * width + c0 ] = left.x;
            result_2y[ r0 * width + c0 ] = left.y; 
        }
        else if (square_case == 4){
            result_1x[ r0 * width + c0 ] = left.x;
            result_1y[ r0 * width + c0 ] = left.y;
            result_2x[ r0 * width + c0 ] = bottom.x;
            result_2y[ r0 * width + c0 ] = bottom.y; 
        }
        else if (square_case == 5){
            result_1x[ r0 * width + c0 ] = top.x;
            result_1y[ r0 * width + c0 ] = top.y;
            result_2x[ r0 * width + c0 ] = bottom.x;
            result_2y[ r0 * width + c0 ] = bottom.y; 
        }
        else if (square_case == 6){
            // TODO !!!
            result_1x[ r0 * width + c0 ] = 0.0;
            result_1y[ r0 * width + c0 ] = 0.0;
            result_2x[ r0 * width + c0 ] = 0.0;
            result_2y[ r0 * width + c0 ] = 0.0; 
            // result_1x[ r0 * width + c0 ] = left.x;
            // result_1y[ r0 * width + c0 ] = left.y;
            // result_2x[ r0 * width + c0 ] = top.x;
            // result_2y[ r0 * width + c0 ] = top.y; 

        }
        else if (square_case == 7){
            result_1x[ r0 * width + c0 ] = right.x;
            result_1y[ r0 * width + c0 ] = right.y;
            result_2x[ r0 * width + c0 ] = bottom.x;
            result_2y[ r0 * width + c0 ] = bottom.y; 
        }
        else if (square_case == 8){
            result_1x[ r0 * width + c0 ] = bottom.x;
            result_1y[ r0 * width + c0 ] = bottom.y;
            result_2x[ r0 * width + c0 ] = right.x;
            result_2y[ r0 * width + c0 ] = right.y; 
        }
        else if (square_case == 9){
            // TODO !!!
            result_1x[ r0 * width + c0 ] = 0.0;
            result_1y[ r0 * width + c0 ] = 0.0;
            result_2x[ r0 * width + c0 ] = 0.0;
            result_2y[ r0 * width + c0 ] = 0.0; 
            // result_1x[ r0 * width + c0 ] = top.x;
            // result_1y[ r0 * width + c0 ] = top.y;
            // result_2x[ r0 * width + c0 ] = left.x;
            // result_2y[ r0 * width + c0 ] = left.y; 
        }
        else if (square_case == 10){
            result_1x[ r0 * width + c0 ] = bottom.x;
            result_1y[ r0 * width + c0 ] = bottom.y;
            result_2x[ r0 * width + c0 ] = top.x;
            result_2y[ r0 * width + c0 ] = top.y; 
        }
        else if (square_case == 11){
            result_1x[ r0 * width + c0 ] = bottom.x;
            result_1y[ r0 * width + c0 ] = bottom.y;
            result_2x[ r0 * width + c0 ] = left.x;
            result_2y[ r0 * width + c0 ] = left.y; 
        }
        else if (square_case == 12){
            result_1x[ r0 * width + c0 ] = left.x;
            result_1y[ r0 * width + c0 ] = left.y;
            result_2x[ r0 * width + c0 ] = right.x;
            result_2y[ r0 * width + c0 ] = right.y; 
        }
        else if (square_case == 13){
            result_1x[ r0 * width + c0 ] = top.x;
            result_1y[ r0 * width + c0 ] = top.y;
            result_2x[ r0 * width + c0 ] = right.x;
            result_2y[ r0 * width + c0 ] = right.y; 
        }
        else if (square_case == 14){
            result_1x[ r0 * width + c0 ] = left.x;
            result_1y[ r0 * width + c0 ] = left.y;
            result_2x[ r0 * width + c0 ] = top.x;
            result_2y[ r0 * width + c0 ] = top.y; 
        }          
    } 
}

void fill(double *input, int w, int h){
    int i, j;
    for (i=0; i<h; i++) {
        for (j=0; j<w; j++) {
            input[i*w+j] = rand() / (RAND_MAX + 1.);
        }
    }
}

int main( int argc, char *argv[] )
{
    double *input;
    double *result_1x, *result_1y, *result_2x, *result_2y; 
    double level = 0.5;
    const size_t n = 48545;
    const size_t w = 511;
    const size_t h = 95;
    const size_t size = n*sizeof(*input);
    const int TIMES = 1000;
    int k = 0;
    
    // alloc the memory for the input on HOST
    input = (double*)malloc(size); assert(input != NULL);
    result_1x = (double*)malloc(size); assert(input != NULL);
    result_1y = (double*)malloc(size); assert(input != NULL);
    result_2x = (double*)malloc(size); assert(input != NULL);
    result_2y = (double*)malloc(size); assert(input != NULL);
    // fill the input
    fill(input, w, h);

    
    double *d_input; /* DEVICE copy */
    double *d_result_1x, *d_result_1y, *d_result_2x, *d_result_2y;  
    const int NUM_BLOCKS_x = (511 + BLKDIM-1) / BLKDIM;
    const int NUM_BLOCKS_y = (95  + BLKDIM-1) / BLKDIM; 
    dim3 block(BLKDIM, BLKDIM);
    dim3 grid(NUM_BLOCKS_x, NUM_BLOCKS_y);

    /* Allocate space for DEVICE copies */
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_result_1x, size);
    cudaMalloc((void **)&d_result_1y, size);
    cudaMalloc((void **)&d_result_2x, size);
    cudaMalloc((void **)&d_result_2y, size);

    // bench the CUDA kernel
    const double tstart = hpc_gettime();

    for(k=0; k<TIMES; k++){
        /* Copy inputs to device */
        cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
        /* Launch marching squares kernel on GPU */
        saxpy<<<grid, block>>>(d_input, d_result_1x, d_result_1y, d_result_2x, d_result_2y, level, n, w, h);
        /* Copy result back to host */
        cudaMemcpy(result_1x, d_result_1x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(result_1y, d_result_1y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(result_2x, d_result_2x, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(result_2y, d_result_2y, size, cudaMemcpyDeviceToHost);
    } 

    const double elapsed = hpc_gettime() - tstart;
    printf("Elapsed time (including data movement): %f\n", elapsed / TIMES);

    // check results
    FILE *fptr;
    fptr = fopen("res1.txt", "w");
    for(int r=0; r<n; r++){
        fprintf(fptr, "\n%f",result_1x[r]);
    }      
    fclose(fptr);

    cudaFree(d_input);
    free(input);

    return EXIT_SUCCESS;
}

/* nvcc bench.cu -o bench -lm */
/* ./bench */