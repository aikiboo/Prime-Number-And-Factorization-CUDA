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
  int threads = NB_THREADS;
  int blocks = (N+NB_THREADS-1)/NB_THREADS;
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
  int threads = NB_THREADS;
  int blocks = (sqrt(N)+NB_THREADS-1)/NB_THREADS;
  (*chrGPU).start();
  isPrimeGPUV1<<<blocks,threads>>>(N,isPrimeArr);
  (*chrGPU).stop();
  cudaMemcpy(isPrime,isPrimeArr ,sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(isPrimeArr);
  return isPrime[0];
}

__global__ void FactorizationGPUV1(const ULONGLONG N,const ULONGLONG* primes,const ULONGLONG primesSize,char* coefs){
    ULONGLONG global_id = blockIdx.x*blockDim.x +threadIdx.x;
    __shared__ char s_coefs[NB_THREADS];
    s_coefs[global_id]=0;
    if(global_id>primesSize)
      return;
    ULONGLONG val = primes[global_id];
    if(N%val){
        char coef =0;
        ULONGLONG tmp = N;
        while(tmp%val==0){
          coef++;
          tmp/=val;
        }
        s_coefs[global_id]=coef;
    }
    coefs[global_id]=s_coefs[global_id];

}

__host__ void FactorizationGPUV1Launcher(const ULONGLONG N,ChronoGPU*chrGPU,vector<ULONGLONG>* primes,vector<Cell> *cells){
  ULONGLONG* primeArr_dev;
  char* coefs_devs;
  char* coefs = (char*)malloc(sizeof(char)*primes->size());
  cudaMalloc(&coefs_devs,sizeof(char)*primes->size());
  cudaMalloc(&primeArr_dev,sizeof(ULONGLONG)*primes->size());
  cudaMemcpy(primeArr_dev,&primes[0] ,sizeof(ULONGLONG)*primes->size(), cudaMemcpyHostToDevice);
  int threads = NB_THREADS;
  int blocks = (N+NB_THREADS-1)/NB_THREADS;
  (*chrGPU).start();
  FactorizationGPUV1<<<blocks,threads>>>(N,primeArr_dev,primes->size(),coefs_devs);
  (*chrGPU).stop();
  cudaMemcpy(coefs,coefs_devs ,sizeof(char)*primes->size(), cudaMemcpyDeviceToHost);
  for(int i =0;i<primes->size();i++){
    if(coefs[i]>0){
      Cell cell;
      cell.expo = coefs[i];
      cell.value = primes->at(i);
      cells->push_back(cell);
    }
  }
}
