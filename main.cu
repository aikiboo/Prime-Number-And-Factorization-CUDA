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



int main(int argc, char const *argv[]) {
  if(argc < 2){
    cout << "Veuillez entrer un nombre à analyser" << endl;
    return 1;
  }
  size_t pos;
  ULONGLONG x = stoll(argv[1],&pos);
  ChronoGPU chrGPU;
	ChronoCPU chrCPU;
  float timeGPU;

  cout<<"======================================"<<endl;
  cout<<"\t Partie CPU sur le nombre "<<x<<endl;
  cout<<"======================================"<<endl;
  cout<<"Test de primalité de "<<x<<endl;
  //Test x is prime
  bool isPrime = isPrimeCPUV1(x,&chrCPU);
  float timeCPU = chrCPU.elapsedTime();
  cout<<"--> Temps du test de primalité : "<<timeCPU<<" ms"<<endl;
  cout<<"Est premier ? "<<(isPrime?"Oui":"Non")<<endl;
  cout<<"Recherche des nombres premiers"<<endl;

  //Find all prime < x
	chrCPU.start();
  vector<ULONGLONG> v = searchPrimesCPUV2(x);
  chrCPU.stop();
  timeCPU = chrCPU.elapsedTime();

  cout<<"--> Temps de recherche : "<<timeCPU<<" ms"<<endl;

  //Factorization of x
  chrCPU.start();
  vector<Cell> cells ={};
  factoCPU(x,&v,&cells);
  chrCPU.stop();
  timeCPU = chrCPU.elapsedTime();

  cout<<"--> Temps de Factorisation : "<<timeCPU<<" ms"<<endl;
  cout<<"Factorisation : 1";
  for(Cell c : cells){
    cout<<" * "<<c.value<<"^"<<c.expo;
  }
  cout<<endl;
  cout<<"======================================"<<endl;
  cout<<"\t Partie GPU sur le nombre "<<x<<endl;
  cout<<"======================================"<<endl;
  cout<<"Test de primalité de "<<x<<endl;
  isPrime = isPrimeGPUlancherV1(x,&chrGPU);
  timeGPU = chrGPU.elapsedTime();
  cout<<"--> Temps du test de primalité : "<<timeGPU<<" ms"<<endl;
  cout<<"Est premier ? "<<(isPrime?"Oui":"Non")<<endl;
  v = searchPrimesGPUV1Launcher(x,&chrGPU);
  timeGPU = chrGPU.elapsedTime();
  cout<<"--> Temps de recherche : "<<timeGPU<<" ms"<<endl;
  vector<Cell> cells_gpu ={};
  factoCPU(x,&v,&cells_gpu);
  timeGPU = chrGPU.elapsedTime();
  cout<<"--> Temps de Factorisation : "<<timeGPU<<" ms"<<endl;
  cout<<"Factorisation : 1";
  for(Cell c : cells_gpu){
    cout<<" * "<<c.value<<"^"<<c.expo;
  }
  cout<<endl;
  return 0;
}
