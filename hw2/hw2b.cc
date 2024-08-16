#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>

int iters, width, height, *image,rank, size,start_j;
double left, right, lower, upper;

void write_png(char *filename, int iters, int width, int height, int *buffer, int step)
{
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
    for (int y = height - 1; y >= 0; --y){
		memset(row, 0, row_size);
		int base = y % size * step + y / size;
		for (int x = 0; x < width; ++x){
			int p = buffer[base * width + x];
			png_bytep color = row + x * 3;
			if (p != iters){
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
void calculatePixels(int *img_){
	double d_x = (right - left) / width;
    double d_y = (upper - lower) / height;
	int repeats;// col;row,
    double x, x0,y, y0, length_squared, temp;
	
	start_j = 0;
    __m128d vec_two = _mm_set_pd1(2);
	__m128d vec_four = _mm_set_pd1(4);
	for (int j = rank; j <height; j += size){
        y0 = j * d_y + lower;
		__m128d y0_vec = _mm_load1_pd(&y0);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < width-1; i+=2){
			
			double x0_2[2] = {i * d_x + left, (i + 1) * d_x + left};
			int repeats2[2] = {0,0};
			bool lock2[2] = {false,false};
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
			img_[start_j * width + i] = repeats2[0];
			img_[start_j * width + i+1] = repeats2[1];              
        }
		if(width%2){
			x0 = (width - 1) * d_x + left;
			repeats = 0;
			x =y= 0;
			//y = 0;
			length_squared = 0;
			while (repeats < iters && length_squared < 4) {
				temp = x * x - y * y + x0;
				y = 2 * x * y + y0;
				x = temp;
				length_squared = x * x + y * y;
				++repeats;
			}
			img_[start_j * width + (width - 1)] = repeats;
		}
        ++start_j;
    }
}

int main(int argc, char **argv)
{
	/* initial MPI*/
	MPI_Group WORLD_GROUP, USED_GROUP;
	MPI_Comm USED_COMM = MPI_COMM_WORLD;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    
	/* argument parsing */
	assert(argc == 9);
	char *filename = argv[1];
	iters = strtol(argv[2], 0, 10);
	left = strtod(argv[3], 0);
	right = strtod(argv[4], 0);
	lower = strtod(argv[5], 0);
	upper = strtod(argv[6], 0);
	width = strtol(argv[7], 0, 10);
	height = strtol(argv[8], 0, 10);

	if (height < size)
	{
		MPI_Comm_group(MPI_COMM_WORLD, &WORLD_GROUP);
		int range[1][3] = {{0, height - 1, 1}};
		MPI_Group_range_incl(WORLD_GROUP, 1, range, &USED_GROUP);
		MPI_Comm_create(MPI_COMM_WORLD, USED_GROUP, &USED_COMM);
		if (USED_COMM == MPI_COMM_NULL)
		{
			MPI_Finalize();
			return 0;
		}
		size = height;
	}

//OpenMP
    int step = ceil((double)height / size);

    int *img_ = (int *)malloc(step * width * sizeof(int));

    calculatePixels(img_);
    

    image = (int *)malloc(size * step * width * sizeof(int));
	MPI_Gather(img_, step * width, MPI_INT, image, step * width, MPI_INT, 0, USED_COMM);

	/* draw and cleanup */
	if (rank == 0){
		write_png(filename, iters, width, height, image, step);
	}

		
	free(img_);
	free(image);
	MPI_Finalize();
	return 0;
}