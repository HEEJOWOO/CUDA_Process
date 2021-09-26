#include "book.h"
#include <iostream>
#include <cuda_runtime.h>
#define N 10
//CPU
void add_cpu(int* a, int* b, int* c) {
	int tid = 0; //0번째 CPU임으로, 0에서 시작
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += 1; //하나의 CPU를 가지고 있으므로, 하나씩 증가
	}
}

//GPU
__global__ void add_gpu(int* a, int* b, int* c) {
	int tid = blockIdx.x;
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

int main(void) {
	printf("cpu compute start\n");
	int a[N], b[N], c[N];

	//CPU에서 배열 'a'와 'b'를 채움
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	add_cpu(a, b, c);

	//결과를 화면에 출력
	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	printf("cpu compute end\n");
	
	printf("gpu compute start\n");
	int* dev_a, * dev_b, * dev_c;

	//GPU 메모리 할당
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	//CPU에서 배열 'a'와 'b'를 채움
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	add_gpu <<<N, 1 >>> (dev_a, dev_b, dev_c);
	
	//배열 'c'를 GPU에서 다시 CPU로 복사
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	//결과 출력
	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	//GPU에 할당된 메모리 해제
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
