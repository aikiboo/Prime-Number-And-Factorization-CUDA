#ifndef __CHRONO_GPU_HPP
#define __CHRONO_GPU_HPP

class ChronoGPU {
private:
	cudaEvent_t m_start;
	cudaEvent_t m_end;
	
	bool m_started;
public:
	ChronoGPU();
	~ChronoGPU();

	void	start();
	void	stop();
	void	reset();
	float	elapsedTime();
};

#endif

