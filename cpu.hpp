#include <iostream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include "utils\chronoCPU.hpp"
#include "commun.hpp"
using namespace std;



bool isPrimeCPUV1(const ULONGLONG N,ChronoCPU*chrCPU);
bool isPrimeCPUV2(const ULONGLONG N,vector<ULONGLONG> &v);
vector<ULONGLONG> searchPrimesCPUV1(const ULONGLONG N);
vector<ULONGLONG> searchPrimesCPUV2(const ULONGLONG N);
void factoCPU(ULONGLONG N,vector<ULONGLONG>* v,vector<Cell>* cells);
