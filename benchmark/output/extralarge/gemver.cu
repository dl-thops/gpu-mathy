#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

__device__ float A[4002][4002];
__device__ float u1[4002];
__device__ float u2[4002];
__device__ float v1[4002];
__device__ float v2[4002];
__device__ float w[4002][4002];
__device__ float x[4002];
__device__ float y[4002];
__device__ float z[4002];

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
	if( !( 0<=j ) || !( j<=(4000-1) ) )return;
	temp_1[j-0] = u1[i] * v1[j] + u2[i] * v2[j];
}

__global__ void kernel_2(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(4000-1) ) )return;
	int thread_count_1 = (4000-1)-0+1;
	float* temp_1 = (float*)malloc(sizeof(float)*((4000-1)-0+1));
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(i,temp_1);
	cudaDeviceSynchronize();
	sumArray( temp_1,thread_count_1);
	cudaDeviceSynchronize();
	A[i][i] = temp_1[0];
}

__global__ void kernel_3(float beta,int i,float* temp_2){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(4000-1) ) )return;
	temp_2[j-0] = beta * A[j][i] * y[j];
}

__global__ void kernel_4(float beta){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(4000-1) ) )return;
	int thread_count_3 = (4000-1)-0+1;
	float* temp_2 = (float*)malloc(sizeof(float)*((4000-1)-0+1));
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>(beta,i,temp_2);
	cudaDeviceSynchronize();
	sumArray( temp_2,thread_count_3);
	cudaDeviceSynchronize();
	x[i] = temp_2[0];
}

__global__ void kernel_5(){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(4000-1) ) )return;
	x[i] = x[i] + z[i];
}

__global__ void kernel_6(float alpha,int i,float* temp_3){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=(4000-1) ) )return;
	temp_3[j-0] = alpha * A[i][j] * x[i];
}

__global__ void kernel_7(float alpha){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(4000-1) ) )return;
	int thread_count_6 = (4000-1)-0+1;
	float* temp_3 = (float*)malloc(sizeof(float)*((4000-1)-0+1));
	kernel_6<<<ceil( 1.0 * thread_count_6/1024),1024>>>(alpha,i,temp_3);
	cudaDeviceSynchronize();
	sumArray( temp_3,thread_count_6);
	cudaDeviceSynchronize();
	w[i][i] = temp_3[0];
}

__global__ void main_kernel(){
	float alpha;
	float beta;
	alpha = 1.5;
	beta = 1.2;
	int thread_count_2 = (4000-1)-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_4 = (4000-1)-0+1;
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>(beta);
	cudaDeviceSynchronize();
	int thread_count_5 = (4000-1)-0+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>();
	cudaDeviceSynchronize();
	int thread_count_7 = (4000-1)-0+1;
	kernel_7<<<ceil( 1.0 * thread_count_7/1024),1024>>>(alpha);
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_A = (float*) malloc(sizeof(float)* (4002)* (4002));
	cudaMemcpyFromSymbol(h_A,A,sizeof(float)* (4002)* (4002));
	float* h_u1 = (float*) malloc(sizeof(float)* (4002));
	cudaMemcpyFromSymbol(h_u1,u1,sizeof(float)* (4002));
	float* h_u2 = (float*) malloc(sizeof(float)* (4002));
	cudaMemcpyFromSymbol(h_u2,u2,sizeof(float)* (4002));
	float* h_v1 = (float*) malloc(sizeof(float)* (4002));
	cudaMemcpyFromSymbol(h_v1,v1,sizeof(float)* (4002));
	float* h_v2 = (float*) malloc(sizeof(float)* (4002));
	cudaMemcpyFromSymbol(h_v2,v2,sizeof(float)* (4002));
	float* h_w = (float*) malloc(sizeof(float)* (4002)* (4002));
	cudaMemcpyFromSymbol(h_w,w,sizeof(float)* (4002)* (4002));
	float* h_x = (float*) malloc(sizeof(float)* (4002));
	cudaMemcpyFromSymbol(h_x,x,sizeof(float)* (4002));
	float* h_y = (float*) malloc(sizeof(float)* (4002));
	cudaMemcpyFromSymbol(h_y,y,sizeof(float)* (4002));
	float* h_z = (float*) malloc(sizeof(float)* (4002));
	cudaMemcpyFromSymbol(h_z,z,sizeof(float)* (4002));
	gettimeofday(&t2, 0);
	double time = 1.0*(t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("Time taken for execution is: %.8f sec\n", time);
	return 0;
}
