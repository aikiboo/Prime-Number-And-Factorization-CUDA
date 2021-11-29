#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <vector>
#include "utils\chronoCPU.hpp"
#include "utils\chronoGPU.hpp"


#include "cpu.hpp"
#include "gpu.hpp"

using namespace std;


void processAndDisplayCPU(ULONGLONG N){
  ChronoCPU chrCPU;
  bool isPrime;
  float timeCPU;
  vector<ULONGLONG> v;
  cout<<"======================================"<<endl;
  cout<<"\t Partie CPU sur le nombre "<<N<<endl;
  cout<<"======================================"<<endl;
  cout<<"Test de primalité de "<<N<<endl;
  //Test N is prime
  isPrime = isPrimeCPUV1(N,&chrCPU);
  timeCPU = chrCPU.elapsedTime();
  cout<<"--> Temps du test de primalité : "<<timeCPU<<" ms"<<endl;
  cout<<"Est premier ? "<<(isPrime?"Oui":"Non")<<endl;
  cout<<"Recherche des nombres premiers"<<endl;
  //Find all prime < N
	chrCPU.start();
  v = searchPrimesCPUV2(N);
  chrCPU.stop();
  timeCPU = chrCPU.elapsedTime();

  cout<<"--> Temps de recherche : "<<timeCPU<<" ms"<<endl;

  //Factorization of N
  chrCPU.start();
  vector<Cell> cells ={};
  factoCPU(N,&v,&cells);
  chrCPU.stop();
  timeCPU = chrCPU.elapsedTime();

  cout<<"--> Temps de Factorisation : "<<timeCPU<<" ms"<<endl;
  cout<<"Factorisation : 1";
  for(Cell c : cells){
    cout<<" * "<<c.value<<"^"<<c.expo;
  }
  cout<<endl;
}

void processAndDisplayGPU(ULONGLONG N){
  ChronoGPU chrGPU;
  bool isPrime;
  float timeGPU;
  vector<ULONGLONG> v;
  cout<<"======================================"<<endl;
  cout<<"\t Partie GPU sur le nombre "<<N<<endl;
  cout<<"======================================"<<endl;
  cout<<"Test de primalité de "<<N<<endl;
  isPrime = isPrimeGPUlancherV1(N,&chrGPU);
  timeGPU = chrGPU.elapsedTime();
  cout<<"--> Temps du test de primalité : "<<timeGPU<<" ms"<<endl;
  cout<<"Est premier ? "<<(isPrime?"Oui":"Non")<<endl;
  v = searchPrimesGPUV1Launcher(N,&chrGPU);
  timeGPU = chrGPU.elapsedTime();
  cout<<"--> Temps de recherche : "<<timeGPU<<" ms"<<endl;
  vector<Cell> cells_gpu ={};
  FactorizationGPUV1Launcher(N,&chrGPU,&v,&cells_gpu);
  timeGPU = chrGPU.elapsedTime();
  cout<<"--> Temps de Factorisation : "<<timeGPU<<" ms"<<endl;
  cout<<"Factorisation : 1";
  for(Cell c : cells_gpu){
    cout<<" * "<<c.value<<"^"<<c.expo;
  }
  cout<<endl;
}



int main(int argc, char const *argv[]) {

  if(argc < 2){
    cout << "Usage : value useCPU(true/false) useGPU(true/false)" << endl;
    return 1;
  }
  size_t pos;
  ULONGLONG x = stoll(argv[1],&pos);
  bool useCPU = true;
  bool useGPU = true;
  if(argc >= 3){

    useCPU = strcmp(argv[2],"true")==0;
    if(argc >=4){
        useGPU = strcmp(argv[3],"true")==0;
    }
  }
  if(useCPU){
    processAndDisplayCPU(x);
  }
  cout<<endl;
  if(useGPU){
    processAndDisplayGPU(x);
  }
  return 0;
}
