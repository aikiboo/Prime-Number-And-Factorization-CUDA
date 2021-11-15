#include "chronoCPU.hpp"

#include <iostream>

ChronoCPU::ChronoCPU() 
	: m_started( false ) {
#ifdef _WIN32
		QueryPerformanceFrequency( &m_frequency );
#endif
}

ChronoCPU::~ChronoCPU() {
	if ( m_started ) {
		stop();
		std::cerr << "ChronoCPU::~ChronoCPU(): chrono wasn't turned off!" << std::endl; 
	}
}

void ChronoCPU::start() {
	if ( !m_started ) {
		m_started = true;
#ifdef _WIN32
		QueryPerformanceCounter( &m_start );
#else
		gettimeofday(&m_start, NULL);
#endif
	}
	else
		std::cerr << "ChronoCPU::start(): chrono wasn't turned off!" << std::endl;
}

void ChronoCPU::stop() {
	if ( m_started ) {
		m_started = false;
#ifdef _WIN32
		QueryPerformanceCounter( &m_stop );
#else
		gettimeofday(&m_stop, NULL);
#endif
	}
	else
		std::cerr << "ChronoCPU::stop(): chrono wasn't started!" << std::endl;
}

float ChronoCPU::elapsedTime() {  
	float time = 0.f;
	if ( m_started ) {
		std::cerr << "ChronoCPU::elapsedTime(): chrono wasn't turned off!" << std::endl;
	}
	else {
#ifdef _WIN32
		time =	( (float)( m_stop.QuadPart - m_start.QuadPart ) / (float)( m_frequency.QuadPart ) ) * 1e3f;
#else
		time = static_cast<float>( m_stop.tv_sec - m_start.tv_sec ) * 1e3f
			+ static_cast<float>(m_stop.tv_usec - m_start.tv_usec ) / 1e3f;
#endif
	}
	return time;
}
