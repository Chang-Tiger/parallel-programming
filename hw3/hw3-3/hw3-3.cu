#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/mman.h>
#include <sys/stat.h> 
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
//======================
#define B 64//block factor of blocked-Floyd Warshall
#define BLOCK_SIZE 32//GPU block size
#define DEV_NO 0
#define INF 0x3FFFFFFF
int vertex, edge, V;
int *Dist = NULL;
__device__ int min_(int a, int b) {return min(a, b);} 
int ceil(int a, int b) { return (a + b - 1) / b; }
__global__ void Phase1(int *dst, int Round, int V){
    int i = threadIdx.y;
	int j = threadIdx.x;
	int k=0;
    int i_B = i + BLOCK_SIZE;
	int j_B = j + BLOCK_SIZE;
    //B為64，一次處理一個64*64，但block dim僅為32*32，因此一個大block分成四個小block，一個thread要計算四個小block四個點
    // 1 2
    // 3 4
	int offset = Round * B * (V+1);
    int blk_pt1 = offset + i * V + j;
    int blk_pt2 = offset + i * V + j_B;
    int blk_pt3 = offset + i_B * V + j;
    int blk_pt4 = offset + i_B * V + j_B;

	__shared__ int share[B][B];
    share[i][j] = dst[blk_pt1];
	share[i][j_B] = dst[blk_pt2];
	share[i_B][j] = dst[blk_pt3];
	share[i_B][j_B] = dst[blk_pt4];
	__syncthreads();
	
	while(k<B){
		share[i][j] = min_(share[i][j], share[i][k] + share[k][j]);
		share[i_B][j] = min_(share[i_B][j], share[i_B][k] + share[k][j]);
		share[i][j_B] = min_(share[i][j_B], share[i][k] + share[k][j_B]);
		share[i_B][j_B] = min_(share[i_B][j_B], share[i_B][k] + share[k][j_B]);
        //if(share[i*B+j]>share[i*B+k] + share[k*B+j]) { share[i*B+j]=share[i*B+k] + share[k*B+j];}
        //if(share[i*B+j_B]>share[i*B+k] + share[k*B+j_B]) { share[i*B+j_B]=share[i*B+k] + share[k*B+j_B];}
        //if(share[i_B*B+j]>share[i_B*B+k] + share[k*B+j]) { share[i_B*B+j]=share[i_B*B+k] + share[k*B+j];}
        //if(share[i_B*B+j_B]>share[i_B*B+k] + share[k*B+j_B]) { share[i_B*B+j_B]=share[i_B*B+k] + share[k*B+j_B];}
		++k;
		__syncthreads();
	}
	dst[blk_pt1] = share[i][j];
	dst[blk_pt2] = share[i][j_B];
	dst[blk_pt3] = share[i_B][j];
	dst[blk_pt4] = share[i_B][j_B];
}

__global__ void Phase2(int *dst, int Round, int V) {
	if (blockIdx.x == Round) {return;}
	
	int i = threadIdx.y;
	int j = threadIdx.x;
	int k=0;
	int offset = Round * B * (V+1);
	int offset_rc, pivot_rc_blk1, pivot_rc_blk2, pivot_rc_blk3, pivot_rc_blk4;
	
	int i_B = i + BLOCK_SIZE;
	int j_B = j + BLOCK_SIZE;
	int blk_pt1 = offset + i * V+ j;
	int blk_pt2 = offset + i * V + j_B;
	int blk_pt3 = offset + i_B * V + j;
	int blk_pt4 = offset + i_B *V + j_B;
	//for same col block
	if(blockIdx.y == 0) {
		offset_rc = blockIdx.x * B * V + Round * B;
		pivot_rc_blk1 = offset_rc + i * V + j;
		pivot_rc_blk2 = offset_rc + i * V + j_B;
		pivot_rc_blk3 = offset_rc + i_B * V + j;
		pivot_rc_blk4 = offset_rc + i_B * V + j_B;
	}else {//for same row block
		offset_rc = Round * B * V + blockIdx.x * B;
		pivot_rc_blk1 = offset_rc + i * V + j;
		pivot_rc_blk2 = offset_rc + i * V + j_B;
		pivot_rc_blk3 = offset_rc + i_B * V + j;
		pivot_rc_blk4 = offset_rc + i_B * V + j_B;
	}

	__shared__ int s[B][B];
	__shared__ int rc_share[B][B];

	s[i][j] = dst[blk_pt1];
	s[i][j_B] = dst[blk_pt2];
	s[i_B][j] = dst[blk_pt3];
	s[i_B][j_B] = dst[blk_pt4];

	rc_share[i][j] = dst[pivot_rc_blk1];
	rc_share[i][j_B] = dst[pivot_rc_blk2];
	rc_share[i_B][j] = dst[pivot_rc_blk3];
	rc_share[i_B][j_B] = dst[pivot_rc_blk4];
	__syncthreads();
	
	if(blockIdx.y == 0){
		while(k<B){
			rc_share[i][j] = min_(rc_share[i][j], rc_share[i][k] + s[k][j]);
			rc_share[i][j_B] = min_(rc_share[i][j_B], rc_share[i][k] + s[k][j_B]);
			rc_share[i_B][j] = min_(rc_share[i_B][j], rc_share[i_B][k] + s[k][j]);
			rc_share[i_B][j_B] = min_(rc_share[i_B][j_B], rc_share[i_B][k] + s[k][j_B]);
			k++;
			__syncthreads();
		}
	} else {
		while(k<B){
			rc_share[i][j] = min_(rc_share[i][j], s[i][k] + rc_share[k][j]);
			rc_share[i][j_B] = min_(rc_share[i][j_B], s[i][k] + rc_share[k][j_B]);
			rc_share[i_B][j] = min_(rc_share[i_B][j], s[i_B][k] + rc_share[k][j]);
			rc_share[i_B][j_B] = min_(rc_share[i_B][j_B], s[i_B][k] + rc_share[k][j_B]);
			k++;
			__syncthreads();
		}
	}

	dst[pivot_rc_blk1] = rc_share[i][j];
	dst[pivot_rc_blk2] = rc_share[i][j_B];
	dst[pivot_rc_blk3] = rc_share[i_B][j];
	dst[pivot_rc_blk4] = rc_share[i_B][j_B];
}

__global__ void Phase3(int *dst, int Round, int V,int r_offset) {
	if (blockIdx.x == Round || blockIdx.y + r_offset == Round) return;
	int i = threadIdx.y;
	int j = threadIdx.x;
	int k=0;
	int i_B = i + BLOCK_SIZE;
	int j_B = j + BLOCK_SIZE;
	
	int offset_ = (blockIdx.y+ r_offset) * B * V + blockIdx.x * B;
	int blk_pt1 = offset_ + i * V + j;
	int blk_pt2 = offset_ + i * V + j_B;
	int blk_pt3 = offset_ + i_B * V + j;
	int blk_pt4 = offset_ + i_B * V + j_B;
	//same row
	int offset_r = (blockIdx.y+ r_offset) * B * V + Round * B;
	int row_blk1 = offset_r + i * V + j;
	int row_blk2 = offset_r + i * V + j_B;
	int row_blk3 = offset_r + i_B * V + j;
	int row_blk4 = offset_r + i_B * V + j_B;
	//same col
	int offset_c = Round * B * V + blockIdx.x * B;
	int col_blk1 = offset_c + i * V + j;
	int col_blk2 = offset_c + i * V+ j_B;
	int col_blk3 = offset_c + i_B * V+ j;
	int col_blk4 = offset_c + i_B * V + j_B;

	

	__shared__ int sh[B][B];
	__shared__ int row_s[B][B];
	__shared__ int col_s[B][B];

	sh[i][j] = dst[blk_pt1];
	sh[i][j_B] = dst[blk_pt2];
	sh[i_B][j] = dst[blk_pt3];
	sh[i_B][j_B] = dst[blk_pt4];

	row_s[i][j] = dst[row_blk1];
	row_s[i][j_B] = dst[row_blk2];
	row_s[i_B][j] = dst[row_blk3];
	row_s[i_B][j_B] = dst[row_blk4];

	col_s[i][j] = dst[col_blk1];
	col_s[i][j_B] = dst[col_blk2];
	col_s[i_B][j] = dst[col_blk3];
	col_s[i_B][j_B] = dst[col_blk4];

	__syncthreads();

		
	while(k<B){
		sh[i][j] = min_(row_s[i][k] + col_s[k][j], sh[i][j]);
		sh[i][j_B] = min_(row_s[i][k] + col_s[k][j_B], sh[i][j_B]);
		sh[i_B][j] = min_(row_s[i_B][k] + col_s[k][j], sh[i_B][j]);
		sh[i_B][j_B] = min_(row_s[i_B][k] + col_s[k][j_B], sh[i_B][j_B]);
		
		//if(sh[i*B+j]>(row_s[i*B+k] + col_s[k*B+j])) {sh[i*B+j]=row_s[i*B+k] + col_s[k*B+j];}
		//if(sh[i*B+j_B]>(row_s[i*B+k] + col_s[k*B+j_B])) {sh[i*B+j_B]=row_s[i*B+k] + col_s[k*B+j_B];}
		//if(sh[i_B*B+j]>(row_s[i_B*B+k] + col_s[k*B+j])) {sh[i_B*B+j]=row_s[i_B*B+k] + col_s[k*B+j];}
		//if(sh[i_B*B+j_B]>(row_s[i_B*B+k] + col_s[k*B+j_B])) {sh[i_B*B+j_B]=row_s[i_B*B+k] + col_s[k*B+j_B];}
		++k;
		//__syncthreads();
	}
	dst[blk_pt1] = sh[i][j];
	dst[blk_pt2] = sh[i][j_B];
	dst[blk_pt3] = sh[i_B][j];
	dst[blk_pt4] = sh[i_B][j_B];
}
void block_FW(int V) {
	int round = ceil(vertex, B);
	int* dst[2];

	cudaHostRegister(Dist, V*V*sizeof(int), cudaHostRegisterDefault);
	
	int blocks = (V + B - 1) / B;
	dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

	#pragma omp parallel num_threads(2)
	{
		// thread number
		unsigned int thread_id = omp_get_thread_num();
		cudaSetDevice(thread_id);
		// neighbor
		unsigned int neighbor_thread_id;
		if(thread_id){
			neighbor_thread_id = 0;
		}else{
			neighbor_thread_id = 1;
		}

		size_t size = V *V* sizeof(int);		
		cudaMalloc(&dst[thread_id], size);
 
		

		unsigned int rounds_to_do = round / 2;
		unsigned int rounds_to_do_neighbor = round / 2;
		if (thread_id == 0) {
			rounds_to_do += round % 2;
		}else{rounds_to_do_neighbor += round % 2;}
		
		dim3 grid_dim(blocks, rounds_to_do);
		dim3 grid_dim2(blocks, 2);
		size_t size_to_do = rounds_to_do * V * B * sizeof(int);
		
		unsigned int start_point;
		if (thread_id) {
			start_point = rounds_to_do_neighbor;
		} else {
			start_point = 0;
		}
		unsigned int _offset = start_point * V * B;

		cudaMemcpy(dst[thread_id] + _offset, Dist + _offset, size_to_do, cudaMemcpyHostToDevice);
		#pragma omp barrier

		for (int r = 0; r < round; ++r) {
			//unsigned int start_point_num = r * B * V;
			if (r >= start_point && r < (start_point + rounds_to_do)) {
				cudaMemcpy(dst[neighbor_thread_id] + r * B * V, dst[thread_id] + r * B * V, B * V * sizeof(int), cudaMemcpyDefault);
			}
			#pragma omp barrier
			// phase 1
			Phase1<<<1, block_dim>>>(dst[thread_id], r, V);
			// phase 2
			Phase2<<<grid_dim2, block_dim>>>(dst[thread_id], r, V);
			// phase 3
			Phase3<<<grid_dim, block_dim>>>(dst[thread_id], r, V, start_point);
		}
		cudaMemcpy(Dist + _offset, dst[thread_id] + _offset, size_to_do, cudaMemcpyDeviceToHost);
	}
	cudaFree(dst[1]);
	cudaFree(dst[0]);
	
}



void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&vertex, sizeof(int), 1, file);
    fread(&edge, sizeof(int), 1, file);
    if (vertex % B){
        V = vertex + (B - vertex % B);//size of total matrix could be filled with B*B blocks
    } else {V = vertex;}
    Dist = (int*)malloc(V*V*sizeof(int));
    //printf("edge:%d vertex:%d V:%d\n",edge,vertex,V);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
                if (i == j){Dist[i*V+j] = 0;}
                else {Dist[i*V+j] = INF;}
                 
            }
    }
    

    int pair[3];
    for (int i = 0; i < edge; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*V+pair[1]] = pair[2];
        //printf("%d ",Dist[pair[0]*V+pair[1]]);
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < vertex; ++i) {
        for (int j = 0; j < vertex; ++j) {
            if (Dist[i*V+j] >= INF) Dist[i*V+j] = INF;
        }
        fwrite(&Dist[i*V], sizeof(int), vertex, outfile);
    }
    fclose(outfile);
}








int main(int argc, char* argv[]) {
    input(argv[1]);
    //cudaGetDeviceProperties(&prop, DEV_NO);
    //printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    block_FW(V);
    output(argv[2]);
    return 0;
}
