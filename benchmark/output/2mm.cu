#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<math.h>
#include <sys/time.h>

__device__ float a[1602][2202];
__device__ float b[2202][1802];
__device__ float c[1802][2402];
__device__ float d[1602][2402];
__device__ float temp[1602][1802];

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

__global__ void kernel_1(float alpha,int i,int j,float* temp_1){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=2200 ) )return;
	temp_1[k-0] = alpha * a[i][k] * b[k][j];
}

__global__ void kernel_2(float alpha,int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=1800 ) )return;
	int thread_count_1 = 2200-0+1;
	float* temp_1 = (float*)malloc(sizeof(float)*(2200-0+1));
	kernel_1<<<ceil( 1.0 * thread_count_1/1024),1024>>>(alpha,i,j,temp_1);
	cudaDeviceSynchronize();
	sumArray( temp_1,thread_count_1);
	cudaDeviceSynchronize();
	temp[i][j] = temp_1[0];
}

__global__ void kernel_3(float alpha){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=(1600-1) ) )return;
	int thread_count_2 = 1800-0+1;
	kernel_2<<<ceil( 1.0 * thread_count_2/1024),1024>>>(alpha,i);
	cudaDeviceSynchronize();
}

__global__ void kernel_4(int i,int j,float* temp_2){
	int k = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=k ) || !( k<=1800 ) )return;
	temp_2[k-0] = temp[i][k] * c[k][j];
}

__global__ void kernel_5(float beta,int i){
	int j = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=j ) || !( j<=2400 ) )return;
	d[i][j] = d[i][j] * beta;
	int thread_count_4 = 1800-0+1;
	float* temp_2 = (float*)malloc(sizeof(float)*(1800-0+1));
	kernel_4<<<ceil( 1.0 * thread_count_4/1024),1024>>>(i,j,temp_2);
	cudaDeviceSynchronize();
	sumArray( temp_2,thread_count_4);
	cudaDeviceSynchronize();
	d[i][j] = temp_2[0];
}

__global__ void kernel_6(float beta){
	int i = 0 + blockDim.x * blockIdx.x + threadIdx.x;
	if( !( 0<=i ) || !( i<=1600 ) )return;
	int thread_count_5 = 2400-0+1;
	kernel_5<<<ceil( 1.0 * thread_count_5/1024),1024>>>(beta,i);
	cudaDeviceSynchronize();
}

__global__ void main_kernel(){
	float alpha;
	float beta;
	alpha = 1.5;
	beta = 1.2;
	int thread_count_3 = (1600-1)-0+1;
	kernel_3<<<ceil( 1.0 * thread_count_3/1024),1024>>>(alpha);
	cudaDeviceSynchronize();
	int thread_count_6 = 1600-0+1;
	kernel_6<<<ceil( 1.0 * thread_count_6/1024),1024>>>(beta);
	cudaDeviceSynchronize();
	return;
}

int main(){
	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	main_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	float* h_a = (float*) malloc(sizeof(float)* (1602)* (2202));
	cudaMemcpyFromSymbol(h_a,a,sizeof(float)* (1602)* (2202));
	float* h_b = (float*) malloc(sizeof(float)* (2202)* (1802));
	cudaMemcpyFromSymbol(h_b,b,sizeof(float)* (2202)* (1802));
	float* h_c = (float*) malloc(sizeof(float)* (1802)* (2402));
	cudaMemcpyFromSymbol(h_c,c,sizeof(float)* (1802)* (2402));
	float* h_d = (float*) malloc(sizeof(float)* (1602)* (2402));
	cudaMemcpyFromSymbol(h_d,d,sizeof(float)* (1602)* (2402));
	float* h_temp = (float*) malloc(sizeof(float)* (1602)* (1802));
	cudaMemcpyFromSymbol(h_temp,temp,sizeof(float)* (1602)* (1802));
	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Time taken for execution is: %.6f ms\n", time);
	return 0;
}
