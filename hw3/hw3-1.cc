#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>
#include <assert.h>
#define INF 0x3FFFFFFF
using namespace std;

void init_dist(vector<vector<int>>& Dist, int &vertex,int &edge, ifstream& input_file){
    #pragma omp parallel for schedule(guided, 1)
    for (int i = 0; i < vertex; ++i) {
        fill(Dist[i].begin(), Dist[i].end(), INF);
        Dist[i][i] = 0;
    }


    //printf("v:%d e:%d\n", vertex, edge);

    int weight,src, dst;
    
    for (int i = 0; i < edge; ++i) {
        input_file.read((char *)&src, sizeof(int));
        input_file.read((char *)&dst, sizeof(int));
		input_file.read((char *)&weight, sizeof(int));
        
        Dist[src][dst] = weight;

    }
}
void floyd_warshall(vector<vector<int>>& Dist, int &vertex){
    for (int k = 0; k < vertex; ++k){
		#pragma omp parallel for schedule(guided, 1) collapse(2)
		for (int i = 0; i < vertex; ++i){
			for (int j = 0; j < vertex; ++j){
				if (Dist[i][j] > Dist[i][k] + Dist[k][j] && Dist[i][k] != INF)
					Dist[i][j] = Dist[i][k] + Dist[k][j];
			}
		}
	}
}
int main(int argc, char **argv)
{

	assert(argc == 3);

	ifstream input_file(argv[1], ios::in | ios::binary);
	ofstream output_file(argv[2], ios::out | ios::binary);
	ios::sync_with_stdio(false);

	int vertex, edge,weight,src, dst;

    
	input_file.read((char *)&vertex, sizeof(int));
	input_file.read((char *)&edge, sizeof(int));


	vector<vector<int>> Dist(vertex, vector<int>(vertex));
    init_dist(Dist, vertex,edge, input_file);
    
	floyd_warshall(Dist,vertex);
	
	
	
    
	for (int i = 0; i < vertex; ++i){
		for (int j = 0; j < vertex; ++j){
			output_file.write((char *)&Dist[i][j], sizeof(int));
		}
	}
	
	return 0;
}