#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>
#include <pthread.h>
int iters, width, height,num_threads, *image;
double left, right, lower, upper;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; //creates mutex variable
int cur_r=0;


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void* calculatePixels(void* arg){

    int start_j=0;//start_j=row
    int i,j,repeats;// col;row,
    double x, x0,y, y0, length_squared, temp;
    double d_x = (right - left) / width;
    double d_y = (upper - lower) / height;
    __m128d vec_two = _mm_set_pd1(2);
	__m128d vec_four = _mm_set_pd1(4);

    
    while(start_j < height){
        pthread_mutex_lock(&mutex);
        if(cur_r < height){
            start_j = cur_r;
            cur_r++;
        }else{ start_j = height;}//go break
        pthread_mutex_unlock(&mutex);
        
        if(start_j < height){
            y0 = start_j * d_y + lower;
            __m128d y0_vec = _mm_load1_pd(&y0);
            for(i=0; i < width; ++i){
                if(i + 1 < width){
                    double x0_2[2] = {i * d_x + left, (i + 1) * d_x + left};
                    int repeats2[2] = {0, 0};
                    bool lock2[2] = {false, false};
                    __m128d x_vec = _mm_set_pd(0, 0);
                    __m128d x0_vec = _mm_load_pd(x0_2);
                    __m128d y_vec = _mm_set_pd(0, 0);
                    __m128d length_squared_vec = _mm_set_pd(0, 0);
                    while (!lock2[0] || !lock2[1]){
                        if (!lock2[0]){
                            if (repeats2[0] < iters && _mm_comilt_sd(length_squared_vec, vec_four)) { ++repeats2[0];}
						    else {lock2[0] =true;}							    
                        }
                        if (!lock2[1]){
                            if (repeats2[1] < iters && _mm_comilt_sd(_mm_shuffle_pd(length_squared_vec, length_squared_vec, 1), vec_four)) { ++repeats2[1];}
                            else { lock2[1] =true;}
                        }
                        __m128d tmp_vec = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x_vec, x_vec), _mm_mul_pd(y_vec, y_vec)), x0_vec);
                        y_vec = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x_vec, y_vec), vec_two), y0_vec);
                        x_vec = tmp_vec;
                        length_squared_vec = _mm_add_pd(_mm_mul_pd(x_vec, x_vec), _mm_mul_pd(y_vec, y_vec));
                    }
                    image[start_j * width + i] = repeats2[0];
                    ++i;
                    image[start_j * width + i] = repeats2[1];  
                }
                else{
                    x0 = i * d_x + left;
                    repeats = 0;
                    x = 0;
                    y = 0;
                    length_squared = 0;
                    while (repeats < iters && length_squared < 4) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[start_j * width + i] = repeats;
                }
               
            }
        }
    }
    
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    
    
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set)*4;

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    //int i;

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    
    pthread_mutex_init(&mutex, NULL);
    pthread_t threads[num_threads];
    //cur_r = 0;
    for(int i=0;i<num_threads;++i){
        pthread_create(&threads[i], NULL, calculatePixels,NULL);
    }

    for(int i=0;i<num_threads;++i){
        pthread_join(threads[i], NULL);
    }


    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}