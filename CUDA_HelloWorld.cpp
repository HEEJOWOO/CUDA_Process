#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>

// 기존 방법
void printHelloWorld()
{
	printf("Hello World!\n");
}

// CUDA 사용 
__global__ void printHelloWorldCUDA()
{
	printf("Hello World from CUDA!\n");
}

int main()
{
	printHelloWorld();
	printHelloWorldCUDA << <1, 1 >> > ();
}
