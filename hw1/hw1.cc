#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#include <string.h>
#include <boost/sort/spreadsort/spreadsort.hpp>



void getBig(float data_prev[],float data1[],float data2[] ,int localSize,int localSize_prev,int &sorted){
    
    int i = localSize-1, j = localSize_prev-1,k=localSize-1;
    while (i>=0&&j>=0&&k >= 0) {
        if (data2[i] >= data_prev[j]) {
            data1[k--] = data2[i--];
        } else {
            data1[k--] = data_prev[j--];
            sorted=1;
        }
    }
    while (i >=0 && k >=0) {
        data1[k--] = data2[i--];
    }
    while (j >=0 && k >=0) {
        data1[k--] = data_prev[j--];
        sorted=1;
    }
}
void getSmall(float data_next[],float data1[],float data2[], int localSize,int localSize_next, int &sorted){
    
    int i = 0, j = 0,k=0;

    while (i < localSize && j < localSize_next && k < localSize) {
        if (data2[i] <= data_next[j]) {
            data1[k++] = data2[i++];
        } else {
            data1[k++] = data_next[j++];
            sorted=1;
        }
    }
    while (i < localSize && k < localSize) {
        data1[k++] = data2[i++];
    }
    while (j < localSize_next && k < localSize) {
        data1[k++] = data_next[j++];
        sorted=1;
    }
}

int main(int argc, char **argv)
{
    double TComm = 0, TIO = 0, Ttemp, TStart;
    
    MPI_Group WORLD_GROUP, USED_GROUP;
	MPI_Comm USED_COMM = MPI_COMM_WORLD;
    int arraySize = atoll(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    MPI_File input_file, output_file;
    
    float *data; // Use an array to store the data
    float *data0;
    float *data_next;
    float *data_prev;
    //int maxsize = size;
    int localSize;
    int localSize_next;
    int localSize_prev;
    int start_x;
    if (argc != 4) {
        fprintf(stderr, "must provide exactly 3 arguments!\n");
        return 1;
    }
    MPI_Init(&argc, &argv);

    
    int rank, size;
    TStart = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    

	

    if(arraySize<size){
        MPI_Comm_group(MPI_COMM_WORLD, &WORLD_GROUP);
		int range[1][3] = {{0, arraySize - 1, 1}};
        MPI_Group_range_incl(WORLD_GROUP, 1, range, &USED_GROUP);
		MPI_Comm_create(MPI_COMM_WORLD, USED_GROUP, &USED_COMM);
        if (USED_COMM == MPI_COMM_NULL){
			MPI_Finalize();
			return 0;
		}
        size = arraySize;
    }
    
    start_x = rank * ((double)arraySize / size);
    int end_x = (rank + 1) * ((double)arraySize / size);
    localSize = end_x - start_x;
    data = new float[localSize]; // Allocate memory for the local data
    data0 = new float[localSize];

    //Calculate the next/prev data size for each process
    if (rank == size - 1) {
        localSize_next = localSize;
    } else {
        int end_next = (rank + 2) * ((double)arraySize / size);
        localSize_next = end_next - end_x;
    }

    if (rank == 0) {
        localSize_prev = localSize;
    } else {
        int start_prev = (rank - 1) * ((double)arraySize / size);
        localSize_prev = start_x - start_prev;
    }
    data_next = new float[localSize_next];
    data_prev = new float[localSize_prev];


    Ttemp = MPI_Wtime();
    MPI_File_open(USED_COMM, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_Offset offset;
    // Calculate the offset for each process
    if(rank < arraySize){
        offset = sizeof(float) * start_x;
        MPI_File_read_at(input_file, offset, data, localSize, MPI_FLOAT, MPI_STATUS_IGNORE);
        
        //std::sort(data, data+localSize);
    }
    

    MPI_File_close(&input_file);
    TIO += MPI_Wtime() - Ttemp;
    boost::sort::spreadsort::spreadsort(data, data+localSize);

    int sorted = 0;
    int allsorted = 1;

    while(allsorted > 0){
        sorted = 0;
        //odd
        if(rank % 2 == 0 && rank == size-1) {memcpy(data0, data, localSize * sizeof(float));}
        if(rank % 2 == 0 && rank != size-1){
            Ttemp = MPI_Wtime();
            MPI_Sendrecv(data, localSize, MPI_FLOAT, rank+1, 0, data_next, localSize_next, MPI_FLOAT, rank+1, 1, USED_COMM, MPI_STATUS_IGNORE);
            TComm += MPI_Wtime() - Ttemp;
            if(data_next[0] > data[localSize-1]){
                memcpy(data0, data, localSize * sizeof(float));
            }else{
                getSmall( data_next,data0,data, localSize,localSize_next,sorted);
            }        
            
        }else if(rank % 2 == 1 && rank != 0){
            Ttemp = MPI_Wtime();
            MPI_Sendrecv(data, localSize, MPI_FLOAT, rank-1, 1, data_prev, localSize_prev, MPI_FLOAT, rank-1, 0, USED_COMM, MPI_STATUS_IGNORE);
            TComm += MPI_Wtime() - Ttemp;
            if(data_prev[localSize_prev-1] < data[0]){
                memcpy(data0, data, localSize * sizeof(float));
            }else{
                getBig(data_prev, data0, data, localSize,localSize_prev, sorted);
            }
        }

        //even
        if(rank == 0) {memcpy(data, data0, localSize * sizeof(float));}
        if(rank % 2 == 1 && rank == size-1) {memcpy(data, data0, localSize * sizeof(float));}
        if(rank % 2 == 1 && rank != size-1){
            Ttemp = MPI_Wtime();
            MPI_Sendrecv(data0, localSize, MPI_FLOAT, rank+1, 0, data_next, localSize_next,MPI_FLOAT, rank+1, 1, USED_COMM, MPI_STATUS_IGNORE);
            TComm += MPI_Wtime() - Ttemp;
            if(data_next[0] > data0[localSize-1]){
                memcpy(data, data0, localSize * sizeof(float));
            }else{
                getSmall(data_next, data, data0, localSize,localSize_next,sorted);  
            }
            
            
        }else if(rank % 2 == 0 && rank != 0){
            Ttemp = MPI_Wtime();
            MPI_Sendrecv(data0, localSize, MPI_FLOAT, rank-1, 1, data_prev, localSize_prev, MPI_FLOAT, rank-1, 0, USED_COMM, MPI_STATUS_IGNORE);
            TComm += MPI_Wtime() - Ttemp;
            if(data_prev[localSize_prev-1] < data0[0]){
                memcpy(data, data0, localSize * sizeof(float));
            }else{
                getBig(data_prev, data, data0, localSize,localSize_prev, sorted);
            }
            
            
        }
        Ttemp = MPI_Wtime();
        MPI_Allreduce(&sorted, &allsorted, 1, MPI_INT, MPI_SUM, USED_COMM);
        TComm += MPI_Wtime() - Ttemp;
    }
    

    Ttemp = MPI_Wtime();
    MPI_File_open(USED_COMM, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

    MPI_File_write_at(output_file, offset, data, localSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    TIO += MPI_Wtime() - Ttemp;
    
    if(rank == 0){
		printf("total time:%lf\ncomputing time:%lf\ncommunication time:%lf\nIO time:%lf\n", MPI_Wtime() - TStart, MPI_Wtime()-TStart-TComm-TIO, TComm, TIO);
	}

    delete[] data, data0, data_next, data_prev; // Free the memory
    
    MPI_Finalize();
    return 0;
}
