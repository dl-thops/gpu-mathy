#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float L[4002][4002];
__device__ float b[4002];
__device__ float x[4002];

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

__global__ void kernel_1(int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(i-1) ) )return;
	x[i] = x[i] - L[i][j] * x[j];
}

__global__ void kernel_2(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(4000-1) ) )return;
	x[i] = b[i];
	int thread_count_1 = (i-1)-0+1;
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(i);
	cudaDeviceSynchronize();
	x[i] = x[i] / L[i][i];
}

__global__ void main_kernel(){
	int thread_count_2 = (4000-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>();
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_L = (float*) malloc(sizeof(float)* (4002)* (4002));
	cudaMemcpyFromSymbol(h_L,L,sizeof(float)* (4002)* (4002));
	float* h_b = (float*) malloc(sizeof(float)* (4002));
	cudaMemcpyFromSymbol(h_b,b,sizeof(float)* (4002));
	float* h_x = (float*) malloc(sizeof(float)* (4002));
	cudaMemcpyFromSymbol(h_x,x,sizeof(float)* (4002));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
