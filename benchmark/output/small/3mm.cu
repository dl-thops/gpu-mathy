#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float a[42][62];
__device__ float ab[42][52];
__device__ float abcd[42][72];
__device__ float b[62][52];
__device__ float c[52][82];
__device__ float cd[52][72];
__device__ float d[82][72];

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
	if( !( 0<=k ) || !( k<=(60-1) ) )return;
	temp_1[k-0] = a[i][k] * b[k][j];
}

__global__ void kernel_2(int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(50-1) ) )return;
	int thread_count_1 = (60-1)-0+1;
	float* temp_1 = (float*)malloc(sizeof(float)*((60-1)-0+1));
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(i,j,temp_1);
	cudaDeviceSynchronize();
	sumArray( temp_1,thread_count_1);
	cudaDeviceSynchronize();
	ab[i][j] = temp_1[0];
}

__global__ void kernel_3(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(40-1) ) )return;
	int thread_count_2 = (50-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void kernel_4(int i,int j,float* temp_2){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(80-1) ) )return;
	temp_2[k-0] = c[i][k] * d[k][j];
}

__global__ void kernel_5(int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(70-1) ) )return;
	int thread_count_4 = (80-1)-0+1;
	float* temp_2 = (float*)malloc(sizeof(float)*((80-1)-0+1));
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>(i,j,temp_2);
	cudaDeviceSynchronize();
	sumArray( temp_2,thread_count_4);
	cudaDeviceSynchronize();
	cd[i][j] = temp_2[0];
}

__global__ void kernel_6(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(50-1) ) )return;
	int thread_count_5 = (70-1)-0+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void kernel_7(int i,int j,float* temp_3){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=(50-1) ) )return;
	temp_3[k-0] = ab[i][k] * cd[k][j];
}

__global__ void kernel_8(int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(70-1) ) )return;
	int thread_count_7 = (50-1)-0+1;
	float* temp_3 = (float*)malloc(sizeof(float)*((50-1)-0+1));
	kernel_7<<<ceil( 1.0 * thread_count_7/1024),1024>>>(i,j,temp_3);
	cudaDeviceSynchronize();
	sumArray( temp_3,thread_count_7);
	cudaDeviceSynchronize();
	abcd[i][j] = temp_3[0];
}

__global__ void kernel_9(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(40-1) ) )return;
	int thread_count_8 = (70-1)-0+1;
	kernel_8<<<ceil( 1.0 * thread_count_8/1024),1024>>>(i);
	cudaDeviceSynchronize();
}

__global__ void main_kernel(){
	int thread_count_3 = (40-1)-0+1;
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_6 = (50-1)-0+1;
	kernel_6<<<ceil( 1.0 * thread_count_6/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_9 = (40-1)-0+1;
	kernel_9<<<ceil( 1.0 * thread_count_9/1024),1024>>>();
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_a = (float*) malloc(sizeof(float)* (42)* (62));
	cudaMemcpyFromSymbol(h_a,a,sizeof(float)* (42)* (62));
	float* h_ab = (float*) malloc(sizeof(float)* (42)* (52));
	cudaMemcpyFromSymbol(h_ab,ab,sizeof(float)* (42)* (52));
	float* h_abcd = (float*) malloc(sizeof(float)* (42)* (72));
	cudaMemcpyFromSymbol(h_abcd,abcd,sizeof(float)* (42)* (72));
	float* h_b = (float*) malloc(sizeof(float)* (62)* (52));
	cudaMemcpyFromSymbol(h_b,b,sizeof(float)* (62)* (52));
	float* h_c = (float*) malloc(sizeof(float)* (52)* (82));
	cudaMemcpyFromSymbol(h_c,c,sizeof(float)* (52)* (82));
	float* h_cd = (float*) malloc(sizeof(float)* (52)* (72));
	cudaMemcpyFromSymbol(h_cd,cd,sizeof(float)* (52)* (72));
	float* h_d = (float*) malloc(sizeof(float)* (82)* (72));
	cudaMemcpyFromSymbol(h_d,d,sizeof(float)* (82)* (72));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
