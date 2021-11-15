
#include "cpu.hpp"

bool isPrimeCPUV1(const ULONGLONG N){
  for(ULONGLONG i = 2;i<sqrt(N);i++){
    if(N%i==0)return false;
  }
  return true;
}

bool isPrimeCPUV2(const ULONGLONG N,vector<ULONGLONG> &v){
  for(ULONGLONG x : v){
    if(x > sqrt(N))
      return true;
    if(N%x==0){
      return false;
    }
  }
  return true;
}


vector<ULONGLONG> searchPrimesCPUV1(const ULONGLONG N){
  std::vector<ULONGLONG> out = {2};
  for(ULONGLONG i=3;i<N;i++){
    bool isPrime = true;
    for(ULONGLONG x : out){
        if(i%x == 0){
          isPrime = false;
          break;
        }
    }
    if(isPrime){
      out.push_back(i);
    }
  }
  return out;
}

vector<ULONGLONG> searchPrimesCPUV2(const ULONGLONG N){
  vector<ULONGLONG> out = {2};
  for(ULONGLONG i=3;i<N;i++){
    if(isPrimeCPUV2(i,out)){
      out.push_back(i);
    }
  }
  return out;
}

void factoCPU(ULONGLONG N,vector<ULONGLONG> v,vector<Cell> &cells){
  ULONGLONG tmp = N;
  for(int i = 0;i<v.size() && tmp!=1;i++){
    ULONGLONG val = v.at(i);
    if(tmp%val==0){
      Cell cell;
      cell.expo = 0;
      cell.value = val;
      while(tmp%val==0){
        cell.expo++;
        tmp/=val;
      }
      cells.push_back(cell);
    }
  }

}
