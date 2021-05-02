#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float A[402][402];
__device__ float x1[402];
__device__ float x2[402];
__device__ float y_1[402];
__device__ float y_2[402];

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

__global__ void kernel_1(int i,float* temp_1){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(400-1) ) )return;
	temp_1[j-0] = A[i][j] * y_1[j];
}

__global__ void kernel_2(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(400-1) ) )return;
	int thread_count_1 = (400-1)-0+1;
	float* temp_1 = (float*)malloc(sizeof(float)*((400-1)-0+1));
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(i,temp_1);
	cudaDeviceSynchronize();
	sumArray( temp_1,thread_count_1);
	cudaDeviceSynchronize();
	x1[i] = temp_1[0];
}

__global__ void kernel_3(int i,float* temp_2){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(400-1) ) )return;
	temp_2[j-0] = A[j][i] * y_2[j];
}

__global__ void kernel_4(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(400-1) ) )return;
	int thread_count_3 = (400-1)-0+1;
	float* temp_2 = (float*)malloc(sizeof(float)*((400-1)-0+1));
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>(i,temp_2);
	cudaDeviceSynchronize();
	sumArray( temp_2,thread_count_3);
	cudaDeviceSynchronize();
	x2[i] = temp_2[0];
}

__global__ void main_kernel(){
	int thread_count_2 = (400-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_4 = (400-1)-0+1;
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>();
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_A = (float*) malloc(sizeof(float)* (402)* (402));
	cudaMemcpyFromSymbol(h_A,A,sizeof(float)* (402)* (402));
	float* h_x1 = (float*) malloc(sizeof(float)* (402));
	cudaMemcpyFromSymbol(h_x1,x1,sizeof(float)* (402));
	float* h_x2 = (float*) malloc(sizeof(float)* (402));
	cudaMemcpyFromSymbol(h_x2,x2,sizeof(float)* (402));
	float* h_y_1 = (float*) malloc(sizeof(float)* (402));
	cudaMemcpyFromSymbol(h_y_1,y_1,sizeof(float)* (402));
	float* h_y_2 = (float*) malloc(sizeof(float)* (402));
	cudaMemcpyFromSymbol(h_y_2,y_2,sizeof(float)* (402));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
