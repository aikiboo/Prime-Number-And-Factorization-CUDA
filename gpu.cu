#include "gpu.hpp"



__global__ void isPrimeGPUV1(const ULONGLONG N,int*isPrime){
  int blockID = blockIdx.x+blockIdx.y*gridDim.x;
  ULONGLONG global_id = blockID* (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) +threadIdx.x;
  while(global_id<N && isPrime[0]==1){
    if(isPrime[0]==1 && global_id>1 && N%global_id==0){
        isPrime[0] = 0;
        return;
    }
    global_id+=blockDim.x*blockDim.y*gridDim.x*gridDim.y;
  }
}

__global__ void isPrimeGPUV2(const ULONGLONG N,int*isPrime,const int* primes,const int sizeofPrimes){
  int blockID = blockIdx.x+blockIdx.y*gridDim.x;
  ULONGLONG global_id = blockID* (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) +threadIdx.x;
  while(global_id<sizeofPrimes){
    if(N%primes[global_id]==0){
      isPrime[0] = 0;
    }
    global_id+=blockDim.x*blockDim.y*gridDim.x*gridDim.y;
  }
}



__host__ vector<ULONGLONG> searchPrimesGPUV1(const ULONGLONG N){
  vector<ULONGLONG> out = {2};
  for(ULONGLONG i=3;i<=N;i++){
    //init primes arrays
    int * primes;
    cudaMalloc(&primes,sizeof(int)*out.size());
    cudaMemcpy(primes,&out[0],sizeof(int),cudaMemcpyHostToDevice);
    //init prime bool
    int* isPrimeArr;
    int isPrime[1] = {1};
    cudaMalloc(&isPrimeArr,sizeof(int));
    cudaMemcpy(isPrimeArr,isPrime,sizeof(int),cudaMemcpyHostToDevice);
    //setup threads
    int threads = 32;
    dim3 nbthread(threads,threads);
    int x = ((out.size()+(threads-1))/threads+(threads-1))/threads;
    dim3 blocks(x,x);
    isPrimeGPUV2<<<blocks,nbthread>>>(i,isPrimeArr,primes,out.size());
    //receive if it's prime
    cudaMemcpy( isPrime,isPrimeArr ,sizeof(bool), cudaMemcpyDeviceToHost);
    if(isPrime[0]){
      //add to std::vector<int> v;
      out.push_back(i);
    }
  }
  return out;
}

__host__ bool isPrimeGPUlancherV1(const ULONGLONG N,ChronoGPU*chrGPU ){
  int* isPrimeArr;
  int isPrime[1] = {1};
  cudaMalloc(&isPrimeArr,sizeof(int));
  cudaMemcpy(isPrimeArr,isPrime,sizeof(int),cudaMemcpyHostToDevice);
  int threads = 32;
  dim3 nbthread(threads,threads);
  int x = ((N+(threads-1))/threads+(threads-1))/threads;
  dim3 blocks(x,x);
  (*chrGPU).start();
  isPrimeGPUV1<<<blocks,nbthread>>>(N,isPrimeArr);
  (*chrGPU).stop();
  cudaMemcpy(isPrime,isPrimeArr ,sizeof(bool), cudaMemcpyDeviceToHost);
  return isPrime[0];
}
