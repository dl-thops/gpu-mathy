#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float A[1002][1202];
__device__ float Q[1002][1202];
__device__ float R[1202][1202];

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

__global__ void kernel_1(int k,float* temp_1){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(1000-1) ) )return;
	temp_1[i-0] = A[i][k] * A[i][k];
}

__global__ void kernel_2(int k){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(1000-1) ) )return;
	Q[i][k] = A[i][k] / R[k][k];
}

__global__ void kernel_3(int j,int k,float* temp_2){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(1000-1) ) )return;
	temp_2[i-0] = Q[i][k] * A[i][j];
}

__global__ void kernel_4(int j,int k){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(1000-1) ) )return;
	A[i][j] = A[i][j] - Q[i][k] * R[k][j];
}

__global__ void kernel_5(int k){
	int j = k + 1 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( k + 1<=j ) || !( j<=(1200-1) ) )return;
	int thread_count_3 = (1000-1)-0+1;
	float* temp_2 = (float*)malloc(sizeof(float)*((1000-1)-0+1));
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>(j,k,temp_2);
	cudaDeviceSynchronize();
	sumArray( temp_2,thread_count_3);
	cudaDeviceSynchronize();
	R[k][j] = temp_2[0];
	int thread_count_4 = (1000-1)-0+1;
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>(j,k);
	cudaDeviceSynchronize();
}

__global__ void kernel_6(float norm_){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(1200-1) ) )return;
	int thread_count_1 = (1000-1)-0+1;
	float* temp_1 = (float*)malloc(sizeof(float)*((1000-1)-0+1));
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(k,temp_1);
	cudaDeviceSynchronize();
	sumArray( temp_1,thread_count_1);
	cudaDeviceSynchronize();
	norm_ = temp_1[0];
	R[k][k] = sqrt(norm_);
	int thread_count_2 = (1000-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>(k);
	cudaDeviceSynchronize();
	int thread_count_5 = (1200-1)-k + 1+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>(k);
	cudaDeviceSynchronize();
}

__global__ void main_kernel(){
	float norm_;
	int thread_count_6 = (1200-1)-0+1;
	kernel_6<<<ceil( 1.0 * thread_count_6/1024),1024>>>(norm_);
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_A = (float*) malloc(sizeof(float)* (1002)* (1202));
	cudaMemcpyFromSymbol(h_A,A,sizeof(float)* (1002)* (1202));
	float* h_Q = (float*) malloc(sizeof(float)* (1002)* (1202));
	cudaMemcpyFromSymbol(h_Q,Q,sizeof(float)* (1002)* (1202));
	float* h_R = (float*) malloc(sizeof(float)* (1202)* (1202));
	cudaMemcpyFromSymbol(h_R,R,sizeof(float)* (1202)* (1202));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
