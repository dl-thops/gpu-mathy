#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float a[6002][8002];
__device__ float ab[6002][7002];
__device__ float abcd[6002][9002];
__device__ float b[8002][7002];
__device__ float c[7002][10002];
__device__ float cd[7002][9002];
__device__ float d[10002][9002];

__global__ void sumCommMultiBlock(float *a, int n) {
	int thIdx = threadIdx.x;
	int gthIdx = thIdx + blockIdx.x*1024;
	const int gridSize = 1024*gridDim.x;
	float sum = 0;
	for (int i = gthIdx; i < n; i += gridSize){
		sum += a[i];
	}
	__shared__ float shArr[1024];
	shArr[thIdx] = sum;
	__syncthreads();
	for (int size = 1024/2; size>0; size/=2) {
		if (thIdx<size){
			shArr[thIdx] += shArr[thIdx+size];
		}
		__syncthreads();
	}
	if (thIdx == 0){
		a[blockIdx.x] = shArr[0];
	}
}

__device__ void sumArray(float* a,int n) {
	sumCommMultiBlock<<<24, 1024>>>(a, n);
	sumCommMultiBlock<<<1, 1024>>>(a, 24);
	cudaDeviceSynchronize();
}

__global__ void prodCommMultiBlock(float *a, int n) {
	int thIdx = threadIdx.x;
	int gthIdx = thIdx + blockIdx.x*1024;
	const int gridSize = 1024*gridDim.x;
	float prod = 1;
	for (int i = gthIdx; i < n; i += gridSize){
		prod *= a[i];
	}
	__shared__ float shArr[1024];
	shArr[thIdx] = prod;
	__syncthreads();
	for (int size = 1024/2; size>0; size/=2) {
		if (thIdx<size){
			shArr[thIdx] *= shArr[thIdx+size];
		}
		__syncthreads();
	}
	if (thIdx == 0){
		a[blockIdx.x] = shArr[0];
	}
}

__device__ void prodArray(float* a,int n) {
	prodCommMultiBlock<<<24, 1024>>>(a, n);
	prodCommMultiBlock<<<1, 1024>>>(a, 24);
	cudaDeviceSynchronize();
}

__global__ void kernel_1(int i,int j,float* temp_1){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(8000-1) ) )return;
	temp_1[k-0] = a[i][k] * b[k][j];
}

__global__ void kernel_2(int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(7000-1) ) )return;
	int thread_count_1 = (8000-1)-0+1;
	float* temp_1 = (float*)malloc(sizeof(float)*((8000-1)-0+1));
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(i,j,temp_1);
	cudaDeviceSynchronize();
	sumArray( temp_1,thread_count_1);
	cudaDeviceSynchronize();
	ab[i][j] = temp_1[0];
}

__global__ void kernel_3(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(6000-1) ) )return;
	int thread_count_2 = (7000-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void kernel_4(int i,int j,float* temp_2){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(10000-1) ) )return;
	temp_2[k-0] = c[i][k] * d[k][j];
}

__global__ void kernel_5(int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(9000-1) ) )return;
	int thread_count_4 = (10000-1)-0+1;
	float* temp_2 = (float*)malloc(sizeof(float)*((10000-1)-0+1));
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>(i,j,temp_2);
	cudaDeviceSynchronize();
	sumArray( temp_2,thread_count_4);
	cudaDeviceSynchronize();
	cd[i][j] = temp_2[0];
}

__global__ void kernel_6(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(7000-1) ) )return;
	int thread_count_5 = (9000-1)-0+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void kernel_7(int i,int j,float* temp_3){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(7000-1) ) )return;
	temp_3[k-0] = ab[i][k] * cd[k][j];
}

__global__ void kernel_8(int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(9000-1) ) )return;
	int thread_count_7 = (7000-1)-0+1;
	float* temp_3 = (float*)malloc(sizeof(float)*((7000-1)-0+1));
	kernel_7<<<ceil( 1.0 * thread_count_7/1024),1024>>>(i,j,temp_3);
	cudaDeviceSynchronize();
	sumArray( temp_3,thread_count_7);
	cudaDeviceSynchronize();
	abcd[i][j] = temp_3[0];
}

__global__ void kernel_9(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(6000-1) ) )return;
	int thread_count_8 = (9000-1)-0+1;
	kernel_8<<<ceil( 1.0 * thread_count_8/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void main_kernel(){
	int thread_count_3 = (6000-1)-0+1;
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_6 = (7000-1)-0+1;
	kernel_6<<<ceil( 1.0 * thread_count_6/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_9 = (6000-1)-0+1;
	kernel_9<<<ceil( 1.0 * thread_count_9/1024),1024>>>();
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_a = (float*) malloc(sizeof(float)* (6002)* (8002));
	cudaMemcpyFromSymbol(h_a,a,sizeof(float)* (6002)* (8002));
	float* h_ab = (float*) malloc(sizeof(float)* (6002)* (7002));
	cudaMemcpyFromSymbol(h_ab,ab,sizeof(float)* (6002)* (7002));
	float* h_abcd = (float*) malloc(sizeof(float)* (6002)* (9002));
	cudaMemcpyFromSymbol(h_abcd,abcd,sizeof(float)* (6002)* (9002));
	float* h_b = (float*) malloc(sizeof(float)* (8002)* (7002));
	cudaMemcpyFromSymbol(h_b,b,sizeof(float)* (8002)* (7002));
	float* h_c = (float*) malloc(sizeof(float)* (7002)* (10002));
	cudaMemcpyFromSymbol(h_c,c,sizeof(float)* (7002)* (10002));
	float* h_cd = (float*) malloc(sizeof(float)* (7002)* (9002));
	cudaMemcpyFromSymbol(h_cd,cd,sizeof(float)* (7002)* (9002));
	float* h_d = (float*) malloc(sizeof(float)* (10002)* (9002));
	cudaMemcpyFromSymbol(h_d,d,sizeof(float)* (10002)* (9002));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}