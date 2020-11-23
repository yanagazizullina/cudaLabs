#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <iostream>
using namespace std;

#define N 1000
#define blocks 32
#define threads 32

__global__ void gpu_calculation_pi(float* estimate, curandState* states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float x, y;
	curand_init(1234, tid, 0, &states[tid]);  
	for (int i = 0; i < N; i++) {
		x = curand_uniform(&states[tid]);
		y = curand_uniform(&states[tid]);
		points_in_circle += (x * x + y * y <= 1.0f); 
	}
	estimate[tid] = 4.0f * points_in_circle / (float)N;
}

float cpu_calculation_pi(long trials) {
	float x, y;
	long points_in_circle = 0.0f;
	for (long i = 0; i < trials; i++) {
		x = rand() / (float)RAND_MAX;
		y = rand() / (float)RAND_MAX;
		points_in_circle += (x * x + y * y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

int main(int argc, char* argv[]) {
	clock_t cpu_start, cpu_stop;
	float host[blocks * threads];
	float* device;
	curandState* deviceStates;
	float gpu_time;
	cudaEvent_t gpu_start, gpu_stop;

	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaMalloc((void**)&device, blocks * threads * sizeof(float)); 
	cudaMalloc((void**)&deviceStates, threads * blocks * sizeof(curandState));
	cudaEventRecord(gpu_start, 0);
	gpu_calculation_pi << <blocks, threads >> > (device, deviceStates);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
	cudaMemcpy(host, device, blocks * threads * sizeof(float), cudaMemcpyDeviceToHost); 
	
	float gpu_pi = 0.0f;
	for (int i = 0; i < blocks * threads; i++) {
		gpu_pi += host[i];
	}

	gpu_pi /= (blocks * threads);

	cout << "Approximate pi calculated on GPU is: " << gpu_pi << " and calculation took " << gpu_time << " msec" << endl;

	cudaFree(device);
	cudaFree(deviceStates);

	cpu_start = clock();
	float cpu_pi = cpu_calculation_pi(blocks * threads * N);
	cpu_stop = clock();
	cout << "Approximate pi calculated on CPU is: " << cpu_pi << " and calculation took " << (cpu_stop - cpu_start) / double(CLOCKS_PER_SEC) * 1000 << " msec" << endl;

	return 0;
}