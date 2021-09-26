# CUDA_Process
  * Visual Studio 2019, CUDA 10.2, GTX 1660 Super 6G
  - 함수 앞에 __global__ 키워드를 추가함으로써 GPU에서 함수를 실행한다고 컴파일러에게 명시
  - GPU 전용 메모리를 사용
    -  cudaMalloc() : 디바이스 메모리 할당
    -  cudaMemcpy() : 디바이스와 호스트 간에 데이터를 복사
    -  cudaFree() : 작업을 마쳤을 시 디바이스 메모리를 해제
