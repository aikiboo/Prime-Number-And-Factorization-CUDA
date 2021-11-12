
#include "cpu.hpp"

bool isPrimeCPU(const ULONGLONG N){
    for(ULONGLONG i = 2;i<sqrt(N);i++){
      if(N%i==0)return false;
    }
    return true;
}
