#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex {
    float   r;
    float   i;
    cuComplex( float a, float b ) : r(a), i(b)  {}
    float magnitude2( void ) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

int julia( int x, int y ) { 
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2); //픽셀 좌표를 복소수 공간으로 변경
    float jy = scale * (float)(DIM/2 - y)/(DIM/2); //픽셀 좌표를 복소수 공간으로 변경

    cuComplex c(-0.8, 0.156); //줄리아 공식에 사용되는 상수 c에 대한 임의이 값, 조정가능
    cuComplex a(jx, jy);

    // 줄리아 공식으로 판별 1000을 넘으면 0 안넘으면 1 반환
    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

void kernel( unsigned char *ptr ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
 }

int main( void ) {
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr = bitmap.get_ptr();

    kernel( ptr );

    bitmap.display_and_exit();
}
