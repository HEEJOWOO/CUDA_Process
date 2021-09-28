/*
* cudaMalloc()를 호출하여 입력 배열 dev_a와 dev_b, 결과 배열 dev_c 이 세 개의 배열 공간을 각각 디바이스에 할당
* cudaFree()를 통한 메모리 해제
* cudaMemcpy()를 이용하되, cudaMemcpyHostToDevice 매개변수를 통해 입력 데이터를 디바이스로 복사, cudaMemcpyDeviceToHost 매개변수를 통해 결과 데이터를 호스트로 복사
* 세 쌍의 꺾쇠괄호 문법을 이용하여 호스트 코드인 main()함수에서 디바이스 코드인 add_gpu()함수를 실행
* 껶쇠 괄호에 대한 추가설명
* <<<N,M>>>는 커널의 실행 방식을 결정하는 런타임 매개 변수, N는 병렬 블록의 개수, 그 수의 블록 만큼 디바이스가 커널을 개시
* <<<2,1>>>는 두 개의 커널 복사본을 생성하고 병렬로 개시한다는 의미 
*/
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
//C 코드를 이용하고 함수 이름에 __global__ 수식 어구를 추가하여 디바이스에서 실행되는 add()함수를 작성
__global__ void add_gpu(int* a, int* b, int* c) {
	int tid = blockIdx.x; //blockIdx는 CUDA 런타임의 내장 변수 중 하나
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
