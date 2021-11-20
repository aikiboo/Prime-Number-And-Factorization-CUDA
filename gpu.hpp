#include <iostream>
#include <vector>
#include <cstdlib>
#include "utils\chronoGPU.hpp"

#define ULONGLONG unsigned long long int
using namespace std;


__global__ void isPrimeGPU(const ULONGLONG N,int*isPrime);

__host__ bool isPrimeGPUlancherV1(const ULONGLONG N,ChronoGPU*chrGPU);
__global__ void searchPrimesGPUV1(const ULONGLONG N,const char* primes);

__host__ vector<ULONGLONG> searchPrimesGPUV1Launcher(const ULONGLONG N,ChronoGPU*chrGPU);
