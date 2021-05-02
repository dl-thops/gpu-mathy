#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float cov_[1202][1202];
__device__ float data_[1402][1202];
__device__ float mean_[1202];

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

__global__ void kernel_1(int x,float* temp_1){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(1400-1) ) )return;
	temp_1[k-0] = data_[k][x] / 1400;
}

__global__ void kernel_2(){
	int x = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=x ) || !( x<=(1200-1) ) )return;
	int thread_count_1 = (1400-1)-0+1;
	float* temp_1 = (float*)malloc(sizeof(float)*((1400-1)-0+1));
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(x,temp_1);
	cudaDeviceSynchronize();
	sumArray( temp_1,thread_count_1);
	cudaDeviceSynchronize();
	mean_[x] = temp_1[0];
}

__global__ void kernel_3(int i,int j,float* temp_2){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(1400-1) ) )return;
	
	
	
	temp_2[k-0] = ((data_[k][i] - mean_[i])*(data_[k][j] - mean_[j]))/2999;
}

__global__ void kernel_4(int i){
	int j = i + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( i<=j ) || !( j<=(1200-1) ) )return;
	int thread_count_3 = (1400-1)-0+1;
	float* temp_2 = (float*)malloc(sizeof(float)*((1400-1)-0+1));
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>(i,j,temp_2);
	cudaDeviceSynchronize();
	sumArray( temp_2,thread_count_3);
	cudaDeviceSynchronize();
	cov_[i][j] = temp_2[0];
	cov_[j][i] = cov_[i][j];
}

__global__ void kernel_5(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(1200-1) ) )return;
	int thread_count_4 = (1200-1)-i+1;
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void main_kernel(){
	int thread_count_2 = (1200-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_5 = (1200-1)-0+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>();
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_cov_ = (float*) malloc(sizeof(float)* (1202)* (1202));
	cudaMemcpyFromSymbol(h_cov_,cov_,sizeof(float)* (1202)* (1202));
	float* h_data_ = (float*) malloc(sizeof(float)* (1402)* (1202));
	cudaMemcpyFromSymbol(h_data_,data_,sizeof(float)* (1402)* (1202));
	float* h_mean_ = (float*) malloc(sizeof(float)* (1202));
	cudaMemcpyFromSymbol(h_mean_,mean_,sizeof(float)* (1202));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
