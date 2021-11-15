#include "chronoGPU.hpp"
#include "common.hpp"
#include <iostream>

ChronoGPU::ChronoGPU() 
	: m_started( false ) {
	HANDLE_ERROR( cudaEventCreate( &m_start ) );
	HANDLE_ERROR( cudaEventCreate( &m_end ) );
}

ChronoGPU::~ChronoGPU() {
	if ( m_started ) {
		stop();
		std::cerr << "ChronoGPU::~ChronoGPU(): hrono wasn't turned off!" << std::endl; 
	}
	HANDLE_ERROR( cudaEventDestroy( m_start ) );
	HANDLE_ERROR( cudaEventDestroy( m_end ) );
}

void ChronoGPU::start() {
	if ( !m_started ) {
		HANDLE_ERROR( cudaEventRecord( m_start, 0 ) );
		m_started = true;
	}
	else
		std::cerr << "ChronoGPU::start(): chrono is already started!" << std::endl;
}

void ChronoGPU::stop() {
	if ( m_started ) {
		HANDLE_ERROR( cudaEventRecord( m_end, 0 ) );
		HANDLE_ERROR( cudaEventSynchronize( m_end ) );
		m_started = false;
	}
	else
		std::cerr << "ChronoGPU::stop(): chrono wasn't started!" << std::endl;
}

float ChronoGPU::elapsedTime() { 
	float time = 0.f;
	HANDLE_ERROR( cudaEventElapsedTime( &time, m_start, m_end ) );
	return time;
}
