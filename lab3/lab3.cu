#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

/* Hint 7 */
// this variable is used by device
__constant__ int mask[MASK_N][MASK_X][MASK_Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};



int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

/* Hint 5 */
// this function is called by host and executed by device
//void sobel (unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels){

    int  x,  i, v, u;
    int  R, G, B;
    double val[MASK_N*3] = {0.0};
    
    int y = threadIdx.x + blockIdx.x * blockDim.x;//tid
    
    if(y > height) {
        return;
    }
    int adjustX, adjustY;
    int xBound = MASK_X /2;
    int yBound = MASK_Y /2;
    double totalR,totalG,totalB;
    // Hint 6 
    // parallel job by blockIdx, blockDim, threadIdx 
    //for (y = 0; y < height; ++y) {
    
    for (x = 0; x < width; ++x) {
        for (i = 0; i < MASK_N; ++i) {
            adjustX = (MASK_X % 2) ? 1 : 0;
            adjustY = (MASK_Y % 2) ? 1 : 0;
            

            val[i*3+2] = 0.0;
            val[i*3+1] = 0.0;
            val[i*3] = 0.0;

            for (v = -yBound; v < yBound + adjustY; ++v) {
                for (u = -xBound; u < xBound + adjustX; ++u) {
                    if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
                        R = s[channels * (width * (y+v) + (x+u)) + 2];
                        G = s[channels * (width * (y+v) + (x+u)) + 1];
                        B = s[channels * (width * (y+v) + (x+u)) + 0];
                        val[i*3+2] += R * mask[i][u + xBound][v + yBound];
                        val[i*3+1] += G * mask[i][u + xBound][v + yBound];
                        val[i*3+0] += B * mask[i][u + xBound][v + yBound];
                    }    
                }
            }
        }

        totalR = 0.0;
        totalG = 0.0;
        totalB = 0.0;
        for (i = 0; i < MASK_N; ++i) {
            totalR += val[i * 3 + 2] * val[i * 3 + 2];
            totalG += val[i * 3 + 1] * val[i * 3 + 1];
            totalB += val[i * 3 + 0] * val[i * 3 + 0];
        }

        totalR = sqrt(totalR) / SCALE;
        totalG = sqrt(totalG) / SCALE;
        totalB = sqrt(totalB) / SCALE;
        const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
        t[channels * (width * y + x) + 2] = cR;
        t[channels * (width * y + x) + 1] = cG;
        t[channels * (width * y + x) + 0] = cB;
    }
    
}

int main(int argc, char** argv) {

    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL;
    unsigned char *dst;
    //unsigned char* host_s = NULL;
    read_png(argv[1], &src, &height, &width, &channels);
    dst = (unsigned char *)malloc(height * width * channels *sizeof(unsigned char));
    //unsigned char* host_t = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));
    
    // Hint 1 
    // cudaMalloc(...) for device src and device dst
    unsigned *d_height, *d_width, *d_channels;
    unsigned char *d_src, *d_dst;
    cudaMalloc((void **)&d_height, sizeof(unsigned));
    cudaMalloc((void **)&d_width, sizeof(unsigned));
    cudaMalloc((void **)&d_channels, sizeof(unsigned));
    cudaMalloc(&d_src, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&d_dst, height * width * channels * sizeof(unsigned char));
    // Hint 2 
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(d_height,&height,sizeof(unsigned),cudaMemcpyHostToDevice);
    cudaMemcpy(d_channels,&channels,sizeof(unsigned),cudaMemcpyHostToDevice);
    cudaMemcpy(d_width,&width,sizeof(unsigned),cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Hint 3 
    // acclerate this function
    //sobel(host_s, host_t, height, width, channels);
    dim3 num_threads(64, 1); // Assuming 64 threads per block
    dim3 num_blocks(((width * height) / num_threads.x) + 1, 1);

    //dim3 num_blocks((width * height + num_threads.x - 1) / num_threads.x, 1);
    

    sobel<<<num_blocks, num_threads>>>(d_src, d_dst, height, width, channels);//<<<num_blocks, num_threads>>>
    // Hint 4
    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, d_dst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    write_png(argv[2], dst, height, width, channels);
    cudaFree(d_height);
    cudaFree(d_width);
    cudaFree(d_channels);
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
