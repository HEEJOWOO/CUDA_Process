#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#ifndef DIM
#define DIM 1000
#endif

struct cuComplex {
               float   r;
               float   i;
               __device__ cuComplex(float a, float b) : r(a), i(b) {}
               __device__ float magnitude2(void) {
                              return r * r + i * i;
               }
               __device__ cuComplex operator*(const cuComplex& a) {
                              return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
               }
               __device__ cuComplex operator+(const cuComplex& a) {
                              return cuComplex(r + a.r, i + a.i);
               }
};

__device__ int julia(int x, int y) {
               const float scale = 1.5;
               float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
               float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

               cuComplex c(-0.8, 0.156);
               cuComplex a(jx, jy);

               int i = 0;
               for (i = 0; i < 200; i++) {
                              a = a * a + c;
                              if (a.magnitude2() > 1000)
                                             return 0;
               }

               return 1;
}

__global__ void kernel(unsigned char* ptr) {
               // map from blockIdx to pixel position
               int x = blockIdx.x;
               int y = blockIdx.y;
               int offset = x + y * gridDim.x;

               // now calculate the value at that position
               int juliaValue = julia(x, y);
               ptr[offset * 4 + 0] = 255 * juliaValue;
               ptr[offset * 4 + 1] = 0;
               ptr[offset * 4 + 2] = 0;
               ptr[offset * 4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
               unsigned char* dev_bitmap;
};

int main(void) {
               DataBlock   data;
               CPUBitmap bitmap(DIM, DIM, &data); //유틸리티 라이브러리를 이용하여 DIM x DIM 사이즈의 비트맵 이지미 생성
               unsigned char* dev_bitmap; //디바이스에서 실행하기 위해 포인터 생성

               HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size())); //데이터를 보관하기 위해 메모리 할당
               data.dev_bitmap = dev_bitmap;

               dim3 grid(DIM, DIM);
               kernel << <grid, 1 >> > (dev_bitmap);

               cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost); // kernel 실행 후 결과값을 다시호스트로 복사

               cudaFree(dev_bitmap);

               bitmap.display_and_exit();
}
