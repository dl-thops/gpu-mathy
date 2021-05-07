#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float A[12002][12002];
__device__ float b[12002];
__device__ float x[12002];
__device__ float y[12002];

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

__global__ void kernel_1(int i,int j,float w){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(j-1) ) )return;
	
	w = w - (A[i][k] * A[k][j]);
}

__global__ void kernel_2(int i,float w){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(i-1) ) )return;
	w = A[i][j];
	int thread_count_1 = (j-1)-0+1;
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(i,j,w);
	cudaDeviceSynchronize();
	A[i][j] = w / A[j][j];
}

__global__ void kernel_3(int i,int j,float w){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(i-1) ) )return;
	
	w = w - (A[i][k] * A[k][j]);
}

__global__ void kernel_4(int i,float w){
	int j = i + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( i<=j ) || !( j<=(12000-1) ) )return;
	w = A[i][j];
	int thread_count_3 = (i-1)-0+1;
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>(i,j,w);
	cudaDeviceSynchronize();
	A[i][j] = w;
}

__global__ void kernel_5(float w){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(12000-1) ) )return;
	int thread_count_2 = (i-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>(i,w);
	cudaDeviceSynchronize();
	int thread_count_4 = (12000-1)-i+1;
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>(i,w);
	cudaDeviceSynchronize();
}

__global__ void kernel_6(int i,float w){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(i-1) ) )return;
	w = w - A[i][j] * y[j];
}

__global__ void kernel_7(float w){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(12000-1) ) )return;
	w = b[i];
	int thread_count_6 = (i-1)-0+1;
	kernel_6<<<ceil( 1.0 * thread_count_6/1024),1024>>>(i,w);
	cudaDeviceSynchronize();
	y[i] = w;
}

__global__ void kernel_8(int i,float w){
	int j = 12000 - i + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 12000 - i<=j ) || !( j<=(12000-1) ) )return;
	w = w - A[11999-i][j] * x[j];
}

__global__ void kernel_9(float w){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(12000-1) ) )return;
	w = y[12000-1-i];
	int thread_count_8 = (12000-1)-12000 - i+1;
	kernel_8<<<ceil( 1.0 * thread_count_8/1024),1024>>>(i,w);
	cudaDeviceSynchronize();
	x[12000-1-i] = w / A[12000-1-i][12000-1-i];
}

__global__ void main_kernel(){
	float w;
	int thread_count_5 = (12000-1)-0+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>(w);
	cudaDeviceSynchronize();
	int thread_count_7 = (12000-1)-0+1;
	kernel_7<<<ceil( 1.0 * thread_count_7/1024),1024>>>(w);
	cudaDeviceSynchronize();
	int thread_count_9 = (12000-1)-0+1;
	kernel_9<<<ceil( 1.0 * thread_count_9/1024),1024>>>(w);
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_A = (float*) malloc(sizeof(float)* (12002)* (12002));
	cudaMemcpyFromSymbol(h_A,A,sizeof(float)* (12002)* (12002));
	float* h_b = (float*) malloc(sizeof(float)* (12002));
	cudaMemcpyFromSymbol(h_b,b,sizeof(float)* (12002));
	float* h_x = (float*) malloc(sizeof(float)* (12002));
	cudaMemcpyFromSymbol(h_x,x,sizeof(float)* (12002));
	float* h_y = (float*) malloc(sizeof(float)* (12002));
	cudaMemcpyFromSymbol(h_y,y,sizeof(float)* (12002));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
