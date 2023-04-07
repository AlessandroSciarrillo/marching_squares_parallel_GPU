extern "C" __device__
float get_fraction(double from_value, double to_value, double level){
    if (to_value == from_value)
        return 0;
    return ((level - from_value) / (to_value - from_value));
}

extern "C" __global__
void saxpy(double *image, double *result_1x, double *result_1y, double *result_2x, double *result_2y, double level, size_t n, size_t width, size_t height)
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

        // skip mask

        double ul = image[ r0 * width + c0 ];
        double ur = image[ r0 * width + c1 ];
        double ll = image[ r1 * width + c0 ];
        double lr = image[ r1 * width + c1 ];

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
            // result_1x[ r0 * width + c0 ] = 0.0;
            // result_1y[ r0 * width + c0 ] = 0.0;
            // result_2x[ r0 * width + c0 ] = 0.0;
            // result_2y[ r0 * width + c0 ] = 0.0; 
            result_1x[ r0 * width + c0 ] = left.x;
            result_1y[ r0 * width + c0 ] = left.y;
            result_2x[ r0 * width + c0 ] = top.x;
            result_2y[ r0 * width + c0 ] = top.y; 

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
            // result_1x[ r0 * width + c0 ] = 0.0;
            // result_1y[ r0 * width + c0 ] = 0.0;
            // result_2x[ r0 * width + c0 ] = 0.0;
            // result_2y[ r0 * width + c0 ] = 0.0; 
            result_1x[ r0 * width + c0 ] = top.x;
            result_1y[ r0 * width + c0 ] = top.y;
            result_2x[ r0 * width + c0 ] = left.x;
            result_2y[ r0 * width + c0 ] = left.y; 
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
