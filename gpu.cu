#include "gpu.hpp"



__global__ void isPrimeGPUV1(const ULONGLONG N,int*isPrime){
  __shared__ int s_isPrime[1];
  ULONGLONG global_id = blockIdx.x +threadIdx.x;
  if(threadIdx.x==0){
    s_isPrime[0]=1;
  }
  __syncthreads();
  while(global_id*global_id<N && s_isPrime[0]==1){
    if(global_id>1 && N%global_id==0){
        s_isPrime[0] = 0;
        break;
    }
    global_id+=blockDim.x*gridDim.x;
  }
  __syncthreads();
  if(threadIdx.x==0 && s_isPrime[0]==0){
    isPrime[0]=s_isPrime[0];
  }
}

__global__ void isPrimeGPUV2(const ULONGLONG N,int*isPrime,const int* primes,const int sizeofPrimes){
  __shared__ int s_isPrime[1];
  ULONGLONG global_id = blockIdx.x +threadIdx.x;
  if(threadIdx.x==0){
    s_isPrime[0]=1;
  }
  __syncthreads();
  while(global_id<sizeofPrimes&&global_id*global_id<N&& s_isPrime[0]==1){
    if(N%primes[global_id]==0){
      isPrime[0] = 0;
      break;
    }
    global_id+=blockDim.x*gridDim.x;
  }
  __syncthreads();
  if(threadIdx.x==0 && s_isPrime[0]==0){
    isPrime[0]=s_isPrime[0];
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
    int blocks = (sqrt(N)+31)/32;
    isPrimeGPUV2<<<blocks,threads>>>(i,isPrimeArr,primes,out.size());
    //receive if it's prime
    cudaMemcpy( isPrime,isPrimeArr ,sizeof(bool), cudaMemcpyDeviceToHost);
    if(isPrime[0]){
      //add to std::vector<int> v;
      out.push_back(i);
    }
    cudaFree(primes);
    cudaFree(isPrimeArr);
  }
  return out;
}

__host__ bool isPrimeGPUlancherV1(const ULONGLONG N,ChronoGPU*chrGPU ){
  int* isPrimeArr;
  int isPrime[1] = {1};
  cudaMalloc(&isPrimeArr,sizeof(int));
  cudaMemcpy(isPrimeArr,isPrime,sizeof(int),cudaMemcpyHostToDevice);
  int threads = 32;
  int blocks = (sqrt(N)+31)/32;
  (*chrGPU).start();
  isPrimeGPUV1<<<blocks,threads>>>(N,isPrimeArr);
  (*chrGPU).stop();
  cudaMemcpy(isPrime,isPrimeArr ,sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(isPrimeArr);
  return isPrime[0];
}
