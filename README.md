# CUDA_Process
  * Visual Studio 2019, CUDA 10.2, GTX 1660 Super 6G
  - 함수 앞에 __global__ 키워드를 추가함으로써 GPU에서 함수를 실행한다고 컴파일러에게 명시
  - GPU 전용 메모리를 사용
    -  cudaMalloc() : 디바이스 메모리 할당
    -  cudaMemcpy() : 디바이스와 호스트 간에 데이터를 복사
    -  cudaFree() : 작업을 마쳤을 시 디바이스 메모리를 해제

 - GPU에서 개시되는 블록들의 집합을 그리드라 함
 - 하나의 그리드는 1또는 2차원의 블록들의 집합이 될 수 있음
 - 커널의 각 복사본은 내장 변수 blockIdx를 통해 어떤 블록에서 현재 실행중인지 판단할 수 있음
   - gridDim : 그리드의 크기를 판단 할 수 있음
 
 - 블록의 수가 하드웨어의 제약을 갖고 있어 블록들을 스레드들로 분할해야함
 - 공유메모리는 블록 내의 스레드들이 계산을 하는데 통신과 협력을 할수 있게함, DRAM에 상주하는 것과 달리 물리적으로 GPU상에 상주
 - CUDA 언어는 상수 메모리로 알려진 또 다른 가용 메모리를 만들어냄, 실행중인 커널에서 변핮 ㅣ않는 데이터를 위해 상수 메모리 사용 
 - 
 
