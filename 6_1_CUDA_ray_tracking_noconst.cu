#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

struct Sphere {
               float   r, b, g;
               float   radius;
               float   x, y, z;
               __device__ float hit(float ox, float oy, float* n) {
                              float dx = ox - x;
                              float dy = oy - y;
                              if (dx * dx + dy * dy < radius * radius) {
                                             float dz = sqrtf(radius * radius - dx * dx - dy * dy);
                                             *n = dz / sqrtf(radius * radius);
                                             return dz + z;
                              }
                              return -INF;
               }
};
#define SPHERES 20


__global__ void kernel(Sphere* s, unsigned char* ptr) {
               // 픽셀 위치 결정
               int x = threadIdx.x + blockIdx.x * blockDim.x;
               int y = threadIdx.y + blockIdx.y * blockDim.y;
               int offset = x + y * blockDim.x * gridDim.x;
               float   ox = (x - DIM / 2);
               float   oy = (y - DIM / 2);

               float   r = 0, g = 0, b = 0;
               float   maxz = -INF;
               for (int i = 0; i < SPHERES; i++) {
                              float   n;
                              float   t = s[i].hit(ox, oy, &n);
                              if (t > maxz) {
                                             float fscale = n;
                                             r = s[i].r * fscale;
                                             g = s[i].g * fscale;
                                             b = s[i].b * fscale;
                                             maxz = t;
                              }
               }

               ptr[offset * 4 + 0] = (int)(r * 255);
               ptr[offset * 4 + 1] = (int)(g * 255);
               ptr[offset * 4 + 2] = (int)(b * 255);
               ptr[offset * 4 + 3] = 255;
}

struct DataBlock {
               unsigned char* dev_bitmap;
               Sphere* s;
};

int main(void) {
               DataBlock   data;
               // 시작 시간 캡처
               cudaEvent_t     start, stop;
               HANDLE_ERROR(cudaEventCreate(&start));
               HANDLE_ERROR(cudaEventCreate(&stop));
               HANDLE_ERROR(cudaEventRecord(start, 0));

               CPUBitmap bitmap(DIM, DIM, &data);
               unsigned char* dev_bitmap;
               Sphere* s;


               // 출력될 비트맵을 위해 GPU 메모리 할당
               HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap,
                              bitmap.image_size()));
               // 구 데이터를 위한 메모리 할당
               HANDLE_ERROR(cudaMalloc((void**)&s,
                              sizeof(Sphere) * SPHERES));

               // 임시 메모리를 할당하고 초기화한 후 GPU의 메모리로 복사한 뒤 임시 메모리 해제
               Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
               for (int i = 0; i < SPHERES; i++) {
                              temp_s[i].r = rnd(1.0f);
                              temp_s[i].g = rnd(1.0f);
                              temp_s[i].b = rnd(1.0f);
                              temp_s[i].x = rnd(1000.0f) - 500;
                              temp_s[i].y = rnd(1000.0f) - 500;
                              temp_s[i].z = rnd(1000.0f) - 500;
                              temp_s[i].radius = rnd(100.0f) + 20;
               }
               //cudaMemcpy를 이용하여 GPU로 구들의 배열을 복사
               HANDLE_ERROR(cudaMemcpy(s, temp_s,
                              sizeof(Sphere) * SPHERES,
                              cudaMemcpyHostToDevice));
               free(temp_s);

               // 구 데이터로부터 비트맵 생성
               dim3    grids(DIM / 16, DIM / 16);
               dim3    threads(16, 16);
               kernel << <grids, threads >> > (s, dev_bitmap);

               // 화면에 출력하기 위해 비트맵을 GPU로부터 다시 복사
               HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost));

               
               HANDLE_ERROR(cudaEventRecord(stop, 0));
               HANDLE_ERROR(cudaEventSynchronize(stop));
               float   elapsedTime;
               HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
                              start, stop));
               printf("Time to generate:  %3.1f ms\n", elapsedTime);

               HANDLE_ERROR(cudaEventDestroy(start));
               HANDLE_ERROR(cudaEventDestroy(stop));

               HANDLE_ERROR(cudaFree(dev_bitmap));
               HANDLE_ERROR(cudaFree(s));

               
               bitmap.display_and_exit();
}
