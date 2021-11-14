
#include "cpu.hpp"

bool isPrimeCPUV1(const ULONGLONG N){
    for(ULONGLONG i = 2;i<sqrt(N);i++){
      if(N%i==0)return false;
    }
    return true;
}
