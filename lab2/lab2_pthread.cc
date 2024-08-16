#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
unsigned long long k = 0;
unsigned long long r = 0;
unsigned long long pixels = 0;
struct ThreadData {
    unsigned long long start;
    unsigned long long end;
    unsigned long long result;
};
void* calculatePixels(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;
    data->result = 0;
    
    for (unsigned long long x = data->start; x < data->end; x++) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        data->result += y;
    }
    data->result %= k;
	pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_lock(&mutex);
    pixels += data->result;
    pthread_mutex_unlock(&mutex);
    pthread_mutex_destroy(&mutex);
	//printf("no:%d\n%d\n",data->start,data->result);
    return NULL;
}
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
	
	unsigned long long numThreads = ncpus*8;
	//printf("numThreads:%d\n",numThreads);
	pthread_t threads[numThreads];
    struct ThreadData threadData[numThreads];
	unsigned long long step = r / numThreads;
	unsigned long long rest = r % numThreads;

	for (unsigned long long i = 0; i < numThreads; i++) {
		if(i < rest){
			threadData[i].start = i * step + i;
			threadData[i].end = i * step + i + step+1;
		}else{
			threadData[i].start = i * step + rest;
			threadData[i].end = i * step + rest + step - 1+1;
		}
		//printf("num:%d start:%d end:%d",i,threadData[i].start, threadData[i].end);
        

        // 創建執行緒，並啟動計算
        pthread_create(&threads[i], NULL, calculatePixels, &threadData[i]);
    }

	for (unsigned long long i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
        //pixels += threadData[i].result;
    }
	printf("%llu\n", (4 * pixels) % k);
}
