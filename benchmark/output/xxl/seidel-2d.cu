#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float A[12003][12003];
__device__ float A_t[12002][12002];

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
	int j = 1 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 1<=j ) || !( j<=(12000 - 2-1) ) )return;
	
	A_t[i][j] = (A[i][j] + A[i-1][j-1] + A[i-1][j] + A[i-1][j+1] + A[i][j-1] + A[i][j+1] + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;
}

__global__ void kernel_2(){
	int i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 1<=i ) || !( i<=(12000 - 2-1) ) )return;
	int thread_count_1 = (12000 - 2-1)-1+1;
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void kernel_3(int i){
	int j = 1 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 1<=j ) || !( j<=(12000 - 2-1) ) )return;
	A[i][j] = A_t[i][j];
}

__global__ void kernel_4(){
	int i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 1<=i ) || !( i<=(12000 - 2-1) ) )return;
	int thread_count_3 = (12000 - 2-1)-1+1;
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void kernel_5(){
	int t = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=t ) || !( t<=(3000-1) ) )return;
	int thread_count_2 = (12000 - 2-1)-1+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_4 = (12000 - 2-1)-1+1;
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>();
	cudaDeviceSynchronize();
}

__global__ void main_kernel(){
	int thread_count_5 = (3000-1)-0+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>();
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_A = (float*) malloc(sizeof(float)* (12003)* (12003));
	cudaMemcpyFromSymbol(h_A,A,sizeof(float)* (12003)* (12003));
	float* h_A_t = (float*) malloc(sizeof(float)* (12002)* (12002));
	cudaMemcpyFromSymbol(h_A_t,A_t,sizeof(float)* (12002)* (12002));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
