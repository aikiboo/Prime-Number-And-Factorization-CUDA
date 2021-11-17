#include <iostream>
#include <vector>
#include <cstdlib>
#include "utils\chronoGPU.hpp"

#define ULONGLONG unsigned long long int
using namespace std;


__global__ void isPrimeGPU(const ULONGLONG N,int*isPrime);

__host__ bool isPrimeGPUlancherV1(const ULONGLONG N,ChronoGPU*chrGPU);

__host__ vector<ULONGLONG> searchPrimesGPUV1(const ULONGLONG N);

__global__ void isPrimeGPUV2(const ULONGLONG N,int*isPrime,const int* primes,const int sizeofPrimes);
