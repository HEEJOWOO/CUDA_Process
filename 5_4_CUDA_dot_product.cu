#include "../common/book.h"

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid =
imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);


__global__ void dot(float* a, float* b, float* c) {
               __shared__ float cache[threadsPerBlock];
               int tid = threadIdx.x + blockIdx.x * blockDim.x;
               int cacheIndex = threadIdx.x;

               float   temp = 0;
               while (tid < N) {
                              temp += a[tid] * b[tid];
                              tid += blockDim.x * gridDim.x;
               }

               // 캐시값 설정
               cache[cacheIndex] = temp;

               // 블록의 스레드들을 동기화
               __syncthreads();

               
               int i = blockDim.x / 2;
               while (i != 0) {
                              if (cacheIndex < i)
                                             cache[cacheIndex] += cache[cacheIndex + i];
                              __syncthreads();
                              i /= 2;
               }

               if (cacheIndex == 0)
                              c[blockIdx.x] = cache[0];
}


int main(void) {
               float* a, * b, c, * partial_c;
               float* dev_a, * dev_b, * dev_partial_c;

               // CPU 메모리 할당
               a = (float*)malloc(N * sizeof(float));
               b = (float*)malloc(N * sizeof(float));
               partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

               // GPU 메모리 할당
               HANDLE_ERROR(cudaMalloc((void**)&dev_a,
                              N * sizeof(float)));
               HANDLE_ERROR(cudaMalloc((void**)&dev_b,
                              N * sizeof(float)));
               HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c,
                              blocksPerGrid * sizeof(float)));

               // 호스트 메모리에 데이터를 채움
               for (int i = 0; i < N; i++) {
                              a[i] = i;
                              b[i] = i * 2;
               }

               // 배열 'a'와 'b'를 GPU로 복사
               HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float),
                              cudaMemcpyHostToDevice));
               HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float),
                              cudaMemcpyHostToDevice));

               dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b,
                              dev_partial_c);

               // 배열 'c'를 GPU에서 CPU로 복사
               HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,
                              blocksPerGrid * sizeof(float),
                              cudaMemcpyDeviceToHost));

               // CPU에서 최종 합 구함
               c = 0;
               for (int i = 0; i < blocksPerGrid; i++) {
                              c += partial_c[i];
               }

#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
               printf("Does GPU value %.6g = %.6g?\n", c,
                              2 * sum_squares((float)(N - 1)));

               // GPU측의 메모리 해제
               HANDLE_ERROR(cudaFree(dev_a));
               HANDLE_ERROR(cudaFree(dev_b));
               HANDLE_ERROR(cudaFree(dev_partial_c));

               // CPU측의 메모리 해제
               free(a);
               free(b);
               free(partial_c);
}
