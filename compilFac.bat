nvcc -o prog main.cu cpu.cpp gpu.cu utils\chronoGPU.cu utils\chronoCPU.cpp -arch=compute_50
