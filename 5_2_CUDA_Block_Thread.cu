#include "../common/book.h"

#define N (33*1024)

__global__ void add(int* a, int* b, int* c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}
int main(void) {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;


	//GPU메모리 할당
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));
	
	//CPU로 배열 'a'와 'b'를 채움
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * i;
	}

	//배열 'a'와 'b'를 GPU로 복사
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int),cudaMemcpyHostToDevice));

	add<<<128, 128 >>>(dev_a, dev_b, dev_c);

	//배열 'c'를 GPU에서 다시 CPU로 복사
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	//요청한 작업을 GPU에서 수행할 수 있는지 확인
	bool success = true;
	for (int i = 0; i < N; i++) {
		if ((a[i] + b[i]) != c[i]) {
			printf("ERROR: %d + %d = %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}
	if (success) printf("We did it!\n");
	
	//GPU에 할당한 메모리 해제
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
