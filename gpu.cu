#include "gpu.hpp"

#include "utils/common.hpp"


__global__ void isPrimeGPUV1(const ULONGLONG N,int*isPrime){
  __shared__ int s_isPrime;
  if(threadIdx.x==0 && (N%2 != 0 || N == 2)){
    s_isPrime = 1;
  }
  __syncthreads();
  ULONGLONG global_id = blockDim.x*blockIdx.x +threadIdx.x;
  ULONGLONG val = global_id*2+3;
  while(s_isPrime  == 1 && val*val<=N){
    if(N%val==0){
        s_isPrime = 0;
        break;
    }
    val+=(blockDim.x*gridDim.x)*2;
  }
  __syncthreads();
  if(threadIdx.x==0 && s_isPrime ==0){
    isPrime[0]=s_isPrime ;
  }
}

__host__ bool isPrimeGPUlancherV1(const ULONGLONG N,ChronoGPU*chrGPU ){
  int* isPrimeArr;
  int isPrime[1] = {1};
  cudaMalloc(&isPrimeArr,sizeof(int));
  cudaMemcpy(isPrimeArr,isPrime,sizeof(int),cudaMemcpyHostToDevice);
  int threads = NB_THREADS;
  int blocks = (sqrt(N)*0.5+NB_THREADS-1)/NB_THREADS;
  (*chrGPU).start();
  isPrimeGPUV1<<<blocks,threads>>>(N,isPrimeArr);
  (*chrGPU).stop();
  cudaMemcpy(isPrime,isPrimeArr ,sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(isPrimeArr);
  return isPrime[0];
}


__global__ void isPrimeGPUV2(const ULONGLONG N,const ULONGLONG*primes,const int primesLen,char*isPrime){
  __shared__ int s_isPrime;
  if(threadIdx.x==0){
    s_isPrime = 1;
  }
  __syncthreads();
  ULONGLONG global_id = blockDim.x*blockIdx.x +threadIdx.x;
  while(s_isPrime == 1 && global_id<primesLen && primes[global_id]){
    if(N%primes[global_id]==0){
      s_isPrime = 0;
      break;
    }
    global_id+=blockDim.x*gridDim.x;
  }
  __syncthreads();
  if(threadIdx.x==0 && s_isPrime ==0){
    isPrime[0]=s_isPrime ;
  }
}

__host__ bool isPrimeGPUlancherV2(const ULONGLONG N,vector<ULONGLONG>* v,ChronoGPU* chrGPU){
  //tableau pour le booléen de primalité
  char* isPrimeArr;
  char isPrime[1] = {1};
  cudaMalloc(&isPrimeArr,sizeof(char));
  cudaMemcpy(isPrimeArr,isPrime,sizeof(char),cudaMemcpyHostToDevice);
  //tableau de premiers
  ULONGLONG* primeArr_dev;
  int nbArrEl = v->size();
  int sizePrimesArr = nbArrEl*sizeof(ULONGLONG);
  ULONGLONG* primesArr = (ULONGLONG*)malloc(sizePrimesArr);
  copy(v->begin(), v->end(), primesArr);
  HANDLE_ERROR(cudaMalloc(&primeArr_dev,sizePrimesArr));
  HANDLE_ERROR(cudaMemcpy(primeArr_dev,primesArr ,sizePrimesArr, cudaMemcpyHostToDevice));

  int threads = NB_THREADS;
  int blocks = (nbArrEl+NB_THREADS-1)/NB_THREADS;
  (*chrGPU).start();
  isPrimeGPUV2<<<blocks,threads>>>(N,primeArr_dev,nbArrEl,isPrimeArr);
  (*chrGPU).stop();
  cudaMemcpy(isPrime,isPrimeArr ,sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(isPrimeArr);
  return isPrime[0];
}





__global__ void searchPrimesGPUV1(const ULONGLONG N,char* primes){
  ULONGLONG global_id = blockIdx.x*blockDim.x +threadIdx.x;
  while(global_id<N){
    ULONGLONG val = (global_id*2)+3;
    if(primes[global_id]==0){
      for(ULONGLONG x=global_id+val;x<N;x+=val){
        primes[x]=1;
      }
    }
    global_id+=blockDim.x*gridDim.x;
  }
}

__host__ vector<ULONGLONG> searchPrimesGPUV1Launcher(const ULONGLONG N,ChronoGPU*chrGPU){
  vector<ULONGLONG> out = {2};
  ULONGLONG nbArrEl = sqrt(N)/2 + 1;
  ULONGLONG sizeArr = sizeof(char)*nbArrEl;
  char* isPrimeArr_dev;
  char* isPrimeArr = (char*) malloc(sizeArr);
  HANDLE_ERROR(cudaMalloc(&isPrimeArr_dev,sizeArr));
  int threads = NB_THREADS;
  int blocks = (nbArrEl+NB_THREADS-1)/NB_THREADS;
  (*chrGPU).start();
  searchPrimesGPUV1<<<blocks,threads>>>(nbArrEl,isPrimeArr_dev);
  (*chrGPU).stop();
  HANDLE_ERROR(cudaMemcpy(isPrimeArr,isPrimeArr_dev ,sizeArr, cudaMemcpyDeviceToHost));
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

    while(global_id<primesSize){
      ULONGLONG val = primes[global_id];
      if(N%val==0){
          char coef =0;
          ULONGLONG tmp = N;
          while(tmp%val==0){
            coef++;
            tmp/=val;
          }
          coefs[global_id]=coef;
      }
      global_id+=blockDim.x*gridDim.x;
    }
}

__host__ void FactorizationGPUV1Launcher(const ULONGLONG N,ChronoGPU*chrGPU,vector<ULONGLONG>* primes,vector<Cell> *cells){
  ULONGLONG* primeArr_dev;
  int nbArrEl = primes->size();
  int sizePrimesArr = nbArrEl*sizeof(ULONGLONG);
  ULONGLONG* primesArr = (ULONGLONG*)malloc(sizePrimesArr);
  int sizeCoefsArr = sizeof(char)*nbArrEl;
  char* coefs_devs;
  char* coefs = (char*)calloc(nbArrEl,sizeof(char));
  copy(primes->begin(), primes->end(), primesArr);
  HANDLE_ERROR(cudaMalloc(&coefs_devs,sizeCoefsArr));
  HANDLE_ERROR(cudaMalloc(&primeArr_dev,sizePrimesArr));
  HANDLE_ERROR(cudaMemcpy(primeArr_dev,primesArr ,sizePrimesArr, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(coefs_devs,coefs ,sizeCoefsArr, cudaMemcpyHostToDevice));
  int threads = NB_THREADS;
  int blocks = (nbArrEl+NB_THREADS-1)/NB_THREADS;
  (*chrGPU).start();
  FactorizationGPUV1<<<blocks,threads>>>(N,primeArr_dev,nbArrEl,coefs_devs);
  (*chrGPU).stop();
  HANDLE_ERROR(cudaMemcpy(coefs,coefs_devs ,sizeCoefsArr, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(primeArr_dev));
  HANDLE_ERROR(cudaFree(coefs_devs));
  for(int i = 0;i<nbArrEl;i++){
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
  ULONGLONG tmp = N;
  for(int j=0;j<cells->size();j++){
      Cell c = cells->at(j);
      for(int i =0;i<c.expo;i++)
        tmp/=c.value;
  }
  if(tmp!=0 && tmp!=1){
    Cell cell;
    cell.expo=1;
    cell.value=tmp;
    cells->push_back(cell);
  }
}
