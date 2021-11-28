#include "gpu.hpp"



__global__ void isPrimeGPUV1(const ULONGLONG N,int*isPrime){
  __shared__ int s_isPrime[1];
  ULONGLONG global_id = blockDim.x*blockIdx.x +threadIdx.x;
  if(threadIdx.x==0){
    s_isPrime[0]=1;
  }
  __syncthreads();
  while( (global_id%2==1 || global_id==2) && s_isPrime[0]==1 && global_id*global_id<=N){
    if(N%global_id==0 && global_id>1 ){
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

__host__ bool isPrimeGPUlancherV1(const ULONGLONG N,ChronoGPU*chrGPU ){
  int* isPrimeArr;
  int isPrime[1] = {1};
  cudaMalloc(&isPrimeArr,sizeof(int));
  cudaMemcpy(isPrimeArr,isPrime,sizeof(int),cudaMemcpyHostToDevice);
  int threads = NB_THREADS;
  int blocks = (sqrt(N)+NB_THREADS-1)/NB_THREADS;
  (*chrGPU).start();
  isPrimeGPUV1<<<1,threads>>>(N,isPrimeArr);
  (*chrGPU).stop();
  cudaMemcpy(isPrime,isPrimeArr ,sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(isPrimeArr);
  return isPrime[0];
}

__global__ void searchPrimesGPUV1(const ULONGLONG N,char* primes){
  ULONGLONG global_id = blockIdx.x*blockDim.x +threadIdx.x;
  ULONGLONG val = (global_id*2)+3;
  if(global_id>N)
    return;
  if(primes[global_id]==0){
    for(ULONGLONG x=global_id+val;x<N;x+=val){
      primes[x]=1;
    }
  }

}

__host__ vector<ULONGLONG> searchPrimesGPUV1Launcher(const ULONGLONG N,ChronoGPU*chrGPU){
  vector<ULONGLONG> out = {2};
  int nbArrEl = sqrt(N)/2 + 1;
  int sizeArr = sizeof(char)*nbArrEl;
  char* isPrimeArr_dev;
  char* isPrimeArr = (char*) malloc(sizeArr);
  cudaMalloc(&isPrimeArr_dev,sizeArr);
  int threads = NB_THREADS;
  int blocks = (nbArrEl+NB_THREADS-1)/NB_THREADS;
  (*chrGPU).start();
  searchPrimesGPUV1<<<blocks,threads>>>(nbArrEl,isPrimeArr_dev);
  (*chrGPU).stop();
  cudaMemcpy(isPrimeArr,isPrimeArr_dev ,sizeArr, cudaMemcpyDeviceToHost);
  cudaFree(isPrimeArr_dev);
  for(ULONGLONG x=0;x<nbArrEl;x++){
    if(!isPrimeArr[x]){
      out.push_back(x*2+3);
    }
  }
  return out;
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
  int nbArrEl = primes->size();
  int sizeArr = sizeof(char)*nbArrEl;
  char* coefs_devs;
  char* coefs = (char*)malloc(sizeArr);
  cudaMalloc(&coefs_devs,sizeArr);
  cudaMalloc(&primeArr_dev,sizeof(ULONGLONG)*nbArrEl);
  cudaMemcpy(primeArr_dev,&primes[0] ,sizeof(ULONGLONG)*nbArrEl, cudaMemcpyHostToDevice);
  int threads = NB_THREADS;
  int blocks = (nbArrEl+NB_THREADS-1)/NB_THREADS;
  (*chrGPU).start();
  FactorizationGPUV1<<<blocks,threads>>>(N,primeArr_dev,nbArrEl,coefs_devs);
  (*chrGPU).stop();
  cudaMemcpy(coefs,coefs_devs ,sizeof(char)*nbArrEl, cudaMemcpyDeviceToHost);
  for(int i =0;i<nbArrEl;i++){
    if(coefs[i]>0){
      Cell cell;
      cell.expo = coefs[i];
      cell.value = primes->at(i);
      cells->push_back(cell);
    }
  }
  if(cells->size()==0){
    Cell cell;
    cell.expo = 1;
    cell.value = N;
    cells->push_back(cell);
  }
}
