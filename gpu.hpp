#ifndef GPU_HPP
#define GPU_HPP
#include <iostream>
#include <vector>
#include <cstdlib>
#include "utils/chronoGPU.hpp"
#include "commun.hpp"
#endif
using namespace std;

#define NB_THREADS 1024


__global__ void isPrimeGPUV1(const ULONGLONG N,int*isPrime);

__host__ bool isPrimeGPUlancherV1(const ULONGLONG N,ChronoGPU*chrGPU);

__global__ void isPrimeGPUV2(const ULONGLONG N,int*isPrime);

__host__ bool isPrimeGPUlancherV2(const ULONGLONG N,vector<ULONGLONG>* v,ChronoGPU*chrGPU);

__global__ void searchPrimesGPUV1(const ULONGLONG N,const char* primes);

__host__ vector<ULONGLONG> searchPrimesGPUV1Launcher(const ULONGLONG N,ChronoGPU*chrGPU);

__global__ void FactorizationGPUV1(const ULONGLONG N,const ULONGLONG* primes,const ULONGLONG primesSize,char* coefs);

__host__ void FactorizationGPUV1Launcher(const ULONGLONG N,ChronoGPU*chrGPU,vector<ULONGLONG>* primes,vector<Cell> *cells);
