#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#define HANDLE_ERROR(_exp) do {											\
    const cudaError_t err = (_exp);										\
    if ( err != cudaSuccess ) {											\
        std::cerr	<< cudaGetErrorString( err ) << " in " << __FILE__	\
					<< " at line " << __LINE__ << std::endl;			\
        exit( EXIT_FAILURE );											\
    }																	\
} while (0)


static unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

static void verifyDimGridBlock( const unsigned int dimGrid, const unsigned int dimBlock,
							    const unsigned int N ) {
	cudaDeviceProp prop;
    int device;
    HANDLE_ERROR(cudaGetDevice(&device));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));

	unsigned long maxGridSize			= prop.maxGridSize[0];
	unsigned long maxThreadsPerBlock	= prop.maxThreadsPerBlock;

	if ( dimBlock > maxThreadsPerBlock ) {
		std::cerr << "Maximum threads per block exceeded: " << dimBlock 
					<< " (max = " << maxThreadsPerBlock << ")" << std::endl;
		exit(EXIT_FAILURE);
	}

	if  ( dimGrid > maxGridSize ) {
		std::cerr << "Maximum grid size exceeded: " << dimGrid 
			<< " (max = " << maxGridSize << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

#endif

