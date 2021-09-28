#include "../common/book.h"

#define N   10

__global__ void add(int* a, int* b, int* c) {
	int tid = threadIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	//GPU 메모리를 할당
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	//CPU로 a,b배열에 값을 채움
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * i;
	}

	//배열 'a'와 'b'를 GPU로 복사
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice));

	add << <1, N >> > (dev_a, dev_b, dev_c);

	//배열 'c'를 GPU에서 다시CPU로 복사
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost));

	//결과 출력
	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	//GPU에 할당된 메모리 해제
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	return 0;
}
