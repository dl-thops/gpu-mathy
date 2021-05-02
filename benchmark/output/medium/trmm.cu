#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float A[202][202];
__device__ float B[202][242];

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
	int k = i + 1 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( i + 1<=k ) || !( k<=(200-1) ) )return;
	temp_1[k-i + 1] = A[k][i] * B[k][j];
}

__global__ void kernel_2(float alpha,int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(240-1) ) )return;
	int thread_count_1 = (200-1)-i + 1+1;
	float* temp_1 = (float*)malloc(sizeof(float)*((200-1)-i + 1+1));
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(i,j,temp_1);
	cudaDeviceSynchronize();
	sumArray( temp_1,thread_count_1);
	cudaDeviceSynchronize();
	B[i][j] = temp_1[0];
	B[i][j] = alpha * B[i][j];
}

__global__ void kernel_3(float alpha){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(200-1) ) )return;
	int thread_count_2 = (240-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>(alpha,i);
	cudaDeviceSynchronize();
}

__global__ void main_kernel(){
	float alpha;
	alpha = 1.5;
	int thread_count_3 = (200-1)-0+1;
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>(alpha);
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_A = (float*) malloc(sizeof(float)* (202)* (202));
	cudaMemcpyFromSymbol(h_A,A,sizeof(float)* (202)* (202));
	float* h_B = (float*) malloc(sizeof(float)* (202)* (242));
	cudaMemcpyFromSymbol(h_B,B,sizeof(float)* (202)* (242));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
