#include "gpu.hpp"

#include "utils/common.hpp"


__global__ void isPrimeGPUV1(const ULONGLONG N,int*isPrime){
  __shared__ int s_isPrime[1];
  ULONGLONG global_id = blockDim.x*blockIdx.x +threadIdx.x;
  ULONGLONG val = global_id*2+3;
  if(threadIdx.x==0 && (N%2 != 0 || N == 2)){
    s_isPrime[0]=1;
  }
  __syncthreads();
  while(s_isPrime[0]==1 && val*val<=N){
    if(N%val==0){
        s_isPrime[0] = 0;
        break;
    }
    val+=(blockDim.x*gridDim.x)*2;
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
    __shared__ char s_coefs[NB_THREADS];
    s_coefs[global_id]=0;
    if(global_id>primesSize)
      return;
    ULONGLONG val = primes[global_id];
    if(N%val==0){
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
  int sizePrimesArr = nbArrEl*sizeof(ULONGLONG);
  ULONGLONG* primesArr = (ULONGLONG*)malloc(sizePrimesArr);
  int sizeCoefsArr = sizeof(char)*nbArrEl;
  char* coefs_devs;
  char* coefs = (char*)malloc(sizeCoefsArr);


  copy(primes->begin(), primes->end(), primesArr);
  HANDLE_ERROR(cudaMalloc(&coefs_devs,sizeCoefsArr));
  HANDLE_ERROR(cudaMalloc(&primeArr_dev,sizePrimesArr));
  HANDLE_ERROR(cudaMemcpy(primeArr_dev,primesArr ,sizePrimesArr, cudaMemcpyHostToDevice));
  int threads = NB_THREADS;
  int blocks = (nbArrEl+NB_THREADS-1)/NB_THREADS;
  //(*chrGPU).start();
  FactorizationGPUV1<<<blocks,threads>>>(N,primeArr_dev,nbArrEl,coefs_devs);
  //(*chrGPU).stop();
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
