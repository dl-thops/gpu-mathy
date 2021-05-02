#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<math.h>
#include <sys/time.h>

__device__ float _fict_[42];
__device__ float ex[62][82];
__device__ float ey[62][82];
__device__ float hz[62][82];

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

__global__ void kernel_1(int t){
	int j = 1 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 1<=j ) || !( j<=(80-1) ) )return;
	ey[0][j] = _fict_[t];
}

__global__ void kernel_2(int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(80-1) ) )return;
	
	ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j]);
}

__global__ void kernel_3(){
	int i = 1 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 1<=i ) || !( i<=(60-1) ) )return;
	int thread_count_2 = (80-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void kernel_4(int i){
	int j = 1 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 1<=j ) || !( j<=(80-1) ) )return;
	
	ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1]);
}

__global__ void kernel_5(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(60-1) ) )return;
	int thread_count_4 = (80-1)-1+1;
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void kernel_6(int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(79-1) ) )return;
	
	hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
}

__global__ void kernel_7(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(59-1) ) )return;
	int thread_count_6 = (79-1)-0+1;
	kernel_6<<<ceil( 1.0 * thread_count_6/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void kernel_8(){
	int t = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=t ) || !( t<=(40-1) ) )return;
	int thread_count_1 = (80-1)-1+1;
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(t);
	cudaDeviceSynchronize();
	int thread_count_3 = (60-1)-1+1;
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_5 = (60-1)-0+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_7 = (59-1)-0+1;
	kernel_7<<<ceil( 1.0 * thread_count_7/1024),1024>>>();
	cudaDeviceSynchronize();
}

__global__ void main_kernel(){
	int thread_count_8 = (40-1)-0+1;
	kernel_8<<<ceil( 1.0 * thread_count_8/1024),1024>>>();
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h__fict_ = (float*) malloc(sizeof(float)* (42));
	cudaMemcpyFromSymbol(h__fict_,_fict_,sizeof(float)* (42));
	float* h_ex = (float*) malloc(sizeof(float)* (62)* (82));
	cudaMemcpyFromSymbol(h_ex,ex,sizeof(float)* (62)* (82));
	float* h_ey = (float*) malloc(sizeof(float)* (62)* (82));
	cudaMemcpyFromSymbol(h_ey,ey,sizeof(float)* (62)* (82));
	float* h_hz = (float*) malloc(sizeof(float)* (62)* (82));
	cudaMemcpyFromSymbol(h_hz,hz,sizeof(float)* (62)* (82));
	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Time taken for execution is: %.6f ms\n", time);
	return 0;
}
