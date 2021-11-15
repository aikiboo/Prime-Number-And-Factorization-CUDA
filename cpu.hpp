#include <iostream>
#include <cstdlib>
#include <math.h>
#include <vector>

#define ULONGLONG unsigned long long int
using namespace std;

struct Cell{
  ULONGLONG value;
  int expo;
};


bool isPrimeCPUV1(const ULONGLONG N);
bool isPrimeCPUV2(const ULONGLONG N,vector<ULONGLONG> &v);
vector<ULONGLONG> searchPrimesCPUV1(const ULONGLONG N);
vector<ULONGLONG> searchPrimesCPUV2(const ULONGLONG N);
void factoCPU(ULONGLONG N,vector<ULONGLONG> v,vector<Cell> &cells);
