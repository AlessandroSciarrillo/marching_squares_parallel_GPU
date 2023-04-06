extern "C" __device__
float get_fraction(float from_value, float to_value, float level){
    if (to_value == from_value)
        return 0;
    return ((level - from_value) / (to_value - from_value));
}

extern "C" __global__
void saxpy(float *image, float *result, size_t n, size_t width, size_t height, float level)
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
    
    if( r0 < height && c0 < width ){
        //result[r0*width+c0] = image[r0*width+c0] +2;

        // skip mask

        float ul = image[ r0 * width + c0 ];
        float ur = image[ r0 * width + c1 ];
        float ll = image[ r1 * width + c0 ];
        float lr = image[ r1 * width + c1 ];

        // skip control for NaN values

        square_case = 0;
        if (ul > level) square_case += 1;
        if (ur > level) square_case += 2;
        if (ll > level) square_case += 4;
        if (lr > level) square_case += 8; 

        if (square_case == 0 || square_case == 15){
            //TODO 
        }

        top.x = r0; 
        top.y = c0 + get_fraction(ul,ur,level);
        bottom.x = r1;
        bottom.y = c0 + get_fraction(ll, lr, level);
        left.x = r0 + get_fraction(ul, ll, level);
        left.y = c0;
        right.x = r0 + get_fraction(ur, lr, level);
        right.y = c1;


        result[r0*width+c0] = square_case;

        if (square_case == 1){

        }
        else if (square_case == 2){

        }


    } 

}
