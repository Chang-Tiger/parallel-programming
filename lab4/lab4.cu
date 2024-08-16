#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8
#define BLOCK_SIZE 32
#define SHA_BLK 36//(BLOCK_SIZE + 4)
#define DEV_NO 0
//cudaDeviceProp prop;

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
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i)
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
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    unsigned offset_y = blockIdx.y * BLOCK_SIZE;
    unsigned offset_x = blockIdx.x * BLOCK_SIZE;
    __shared__ unsigned char share[SHA_BLK * SHA_BLK * 3];
    int tid = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    for (int i = tid; i < SHA_BLK * SHA_BLK * channels; i += BLOCK_SIZE * BLOCK_SIZE) {
        int cha = i % channels;
        int col = (i / channels) % SHA_BLK;
        int row = (i / channels) / SHA_BLK;
        if (bound_check(offset_x + col - xBound, 0, width) && bound_check(offset_y + row - yBound, 0, height)) {
            share[i] = s[channels * (width * (offset_y + row - yBound) + (offset_x + col - xBound)) + cha];
        }
    }
    __syncthreads();

    unsigned x = offset_x + threadIdx.x;
    if (x >= width) return;

    unsigned y = offset_y + threadIdx.y;
    if (y >= height) return;

    /* Y and X axis of mask */
    int val00 = 0; int val01 = 0; int val02 = 0;
    int val10 = 0; int val11 = 0; int val12 = 0;
    for (int v = -yBound; v <= yBound; ++v) {
        for (int u = -xBound; u <= xBound; ++u) {
            if (bound_check(x + u, 0, width) && bound_check(y + v, 0, height)) {
                int yi = threadIdx.y + v + yBound;
                int xi = threadIdx.x + u + xBound;
                const unsigned char R = share[channels * (SHA_BLK * yi + xi) + 2];
                const unsigned char G = share[channels * (SHA_BLK * yi + xi) + 1];
                const unsigned char B = share[channels * (SHA_BLK * yi + xi) + 0];
                val02 += R * mask[0][u + xBound][v + yBound];
                val01 += G * mask[0][u + xBound][v + yBound];
                val00 += B * mask[0][u + xBound][v + yBound];
                val12 += R * mask[1][u + xBound][v + yBound];
                val11 += G * mask[1][u + xBound][v + yBound];
                val10 += B * mask[1][u + xBound][v + yBound];
            }
        }
    }

    float totalR = val02 * val02 + val12 * val12;
    float totalG = val01 * val01 + val11 * val11;
    float totalB = val00 * val00 + val10 * val10;

    totalR = sqrtf(totalR) / SCALE;
    totalG = sqrtf(totalG) / SCALE;
    totalB = sqrtf(totalB) / SCALE;
    unsigned char cR = (totalR > 255.) ? 255 : totalR;
    unsigned char cG = (totalG > 255.) ? 255 : totalG;
    unsigned char cB = (totalB > 255.) ? 255 : totalB;
    t[channels * (width * y + x) + 2] = cR;
    t[channels * (width * y + x) + 1] = cG;
    t[channels * (width * y + x) + 0] = cB;
    
}

int main(int argc, char** argv) {
    assert(argc == 3);
    //cudaGetDeviceProperties(&prop, DEV_NO);
    //printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    unsigned height, width, channels;
    unsigned char* src = NULL, * dst;
    unsigned char* dsrc, * ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    dst = (unsigned char*)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);
    cudaHostRegister(dst, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    const int b_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int b_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_dim(b_x, b_y);
    sobel << <grid_dim, block_dim >> > (dsrc, ddst, height, width, channels);


    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}

