#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);

    unsigned long long pixels = 0;
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned long long start_x = rank * ((double)r / size);
    unsigned long long end_x = (rank + 1) * ((double)r / size);

    for (unsigned long long x = start_x; x < end_x; x++) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        pixels += y;
        
    }
    pixels %= k;
    unsigned long long total_pixels = 0;
    MPI_Reduce(&pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("%llu\n", (4 * total_pixels) % k);
    }

    MPI_Finalize();

    return 0;
}
