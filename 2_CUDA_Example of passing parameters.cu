/*
* HANDLE_ERROR()는 유틸리티 매크로 함수, 해당 호출이 오류를 반환하는지 점검, 만약 오류가 ㅂ라생했다면 관련 오류 메시지를 출력 후 EXIT_FAILURE코드와 함께 어플리케이션 종료
* 본 프로젝트의 목표
* cudaMalloc()으로 할당한 메모리 포인터를 디바이스에서 실행되는 함수로 전달
* 디바이스에서 실행되는 코드에서 cudaMalloc()로 할당한 메모리 포인터를 이요하여 메모리를 읽거나 씀
* cudaMalloc()으로 할당한 메모리 포인터를 호스트에서 실행되는 함수로 전달할 수 있음
* 호스트에서 실행되는 코드에서 cudaMalloc()으로 할당한 메모리 포인터를 이용하여 메모리를 읽거나 쓸 수 없음
*/

#include <iostream>
#include <cuda_runtime.h>
#include "book.h"
__global__ void add(int a, int b, int* c, int* d) {
	*c = a + b;
	*d = a * b;
}

int main(void) {
	int c;
	int *dev_c;
	int d;
	int* dev_d;
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_d, sizeof(int)));
	add << <1, 1 >> > (2, 7, dev_c,dev_d);
	HANDLE_ERROR(cudaMemcpy(&c,
		dev_c,
		sizeof(int),
		cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&d,
		dev_d,
		sizeof(int),
		cudaMemcpyDeviceToHost));
	printf("2+7=%d\n", c);
	printf("2*7=%d\n", d);
	cudaFree(dev_c);
	cudaFree(dev_d);
	return 0;
}
