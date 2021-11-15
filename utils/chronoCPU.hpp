#ifndef __CHRONO_CPU_HPP
#define __CHRONO_CPU_HPP

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

class ChronoCPU {
private:
#ifdef _WIN32
	LARGE_INTEGER m_frequency;
	LARGE_INTEGER m_start;
	LARGE_INTEGER m_stop;
#else
	timeval m_start;
	timeval m_stop;
#endif

	bool m_started;

public:
	ChronoCPU();
	~ChronoCPU();

	void	start();
	void	stop();
	float	elapsedTime();
};

#endif

