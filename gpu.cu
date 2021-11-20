#include "gpu.hpp"



__global__ void isPrimeGPUV1(const ULONGLONG N,int*isPrime){
  __shared__ int s_isPrime[1];
  ULONGLONG global_id = blockDim.x*blockIdx.x +threadIdx.x;
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

__global__ void searchPrimesGPUV1(const ULONGLONG N,char* primes){
  ULONGLONG global_id = blockIdx.x*blockDim.x +threadIdx.x;
  ULONGLONG val = (global_id*2)+3;
  if(global_id>N/2)
    return;
  primes[global_id]=1;
  __syncthreads();

  if(primes[global_id]==1){
    for(ULONGLONG x=global_id+val;x<N/2;x+=val){
      primes[x]=0;
    }
  }

}

__host__ vector<ULONGLONG> searchPrimesGPUV1Launcher(const ULONGLONG N,ChronoGPU*chrGPU){
  vector<ULONGLONG> out = {2};
  char* isPrimeArr_dev;
  char* isPrimeArr = (char*) malloc(sizeof(char)*N/2);
  cudaMalloc(&isPrimeArr_dev,sizeof(char)*N/2);
  int threads = 32;
  int blocks = (N+31)/32;
  (*chrGPU).start();
  searchPrimesGPUV1<<<blocks,threads>>>(N,isPrimeArr_dev);
  (*chrGPU).stop();
  cudaMemcpy(isPrimeArr,isPrimeArr_dev ,sizeof(char)*N/2, cudaMemcpyDeviceToHost);
  cudaFree(isPrimeArr_dev);
  for(ULONGLONG x=0;x<N/2;x++){
    if(isPrimeArr[x]){
      out.push_back(x*2+3);
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
  int blocks = (sqrt(N)+31)/32;
  (*chrGPU).start();
  isPrimeGPUV1<<<blocks,threads>>>(N,isPrimeArr);
  (*chrGPU).stop();
  cudaMemcpy(isPrime,isPrimeArr ,sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(isPrimeArr);
  return isPrime[0];
}
