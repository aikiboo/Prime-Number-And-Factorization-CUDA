#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <vector>

#include "cpu.hpp"

using namespace std;



int main(int argc, char const *argv[]) {
  if(argc < 2){
    cout << "Veuillez entrer un nombre à analyser" << endl;
    return 1;
  }
  size_t pos;
  ULONGLONG x = stoll(argv[1],&pos);
  cout <<x<<endl;
  cout<<"Is prime : "<<isPrimeCPUV1(x)<<endl;

  vector<ULONGLONG> v = searchPrimesCPUV2(x);
  for (ULONGLONG n : v) {
        std::cout << n << ", ";
    }
  return 0;
}
