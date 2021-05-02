#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float data_[102][102];
__device__ float mean_[102];
__device__ float std_dev_[102];

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
	if( !( 0<=k ) || !( k<=100 ) )return;
	temp_1[k-0] = data_[k][x] / 100;
}

__global__ void kernel_2(){
	int x = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=x ) || !( x<=100 ) )return;
	int thread_count_1 = 100-0+1;
	float* temp_1 = (float*)malloc(sizeof(float)*(100-0+1));
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(x,temp_1);
	cudaDeviceSynchronize();
	sumArray( temp_1,thread_count_1);
	cudaDeviceSynchronize();
	mean_[x] = temp_1[0];
}

__global__ void kernel_3(int t,float* temp_2){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=100 ) )return;
	
	
	
	temp_2[k-0] = ((data_[k][t] - mean_[t])*(data_[k][t] - mean_[t]))/99;
}

__global__ void kernel_4(){
	int t = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=t ) || !( t<=100 ) )return;
	int thread_count_3 = 100-0+1;
	float* temp_2 = (float*)malloc(sizeof(float)*(100-0+1));
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>(t,temp_2);
	cudaDeviceSynchronize();
	sumArray( temp_2,thread_count_3);
	cudaDeviceSynchronize();
	std_dev_[t] = temp_2[0];
}

__global__ void kernel_5(){
	int p = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=p ) || !( p<=100 ) )return;
	std_dev_[p] = sqrt(std_dev_[p]);
}

__global__ void main_kernel(){
	int thread_count_2 = 100-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_4 = 100-0+1;
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_5 = 100-0+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>();
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_data_ = (float*) malloc(sizeof(float)* (102)* (102));
	cudaMemcpyFromSymbol(h_data_,data_,sizeof(float)* (102)* (102));
	float* h_mean_ = (float*) malloc(sizeof(float)* (102));
	cudaMemcpyFromSymbol(h_mean_,mean_,sizeof(float)* (102));
	float* h_std_dev_ = (float*) malloc(sizeof(float)* (102));
	cudaMemcpyFromSymbol(h_std_dev_,std_dev_,sizeof(float)* (102));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
