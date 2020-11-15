#include <iostream>
#include <algorithm>
#include <ctime>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define BLOCKDIM 32

__constant__ float cGaussian[64];
texture<unsigned char, 2, cudaReadModeElementType> inTexture;

void updateGaussian(int r, double sd)
{
	float fGaussian[64];
	for (int i = 0; i < 2 * r + 1; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x * x) / (2 * sd * sd));
	}
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float) * (2 * r + 1));
}


__device__ inline double gaussian(float x, double sigma)
{
	return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

__global__ void gpuCalculation(unsigned char* input, unsigned char* output,
	int width, int height,
	int r, double sI, double sS)
{
	int txIndex = __mul24(blockIdx.x, BLOCKDIM) + threadIdx.x;
	int tyIndex = __mul24(blockIdx.y, BLOCKDIM) + threadIdx.y;

	if ((txIndex < width) && (tyIndex < height))
	{
		double iFiltered = 0;
		double wP = 0;
		unsigned char centrePx = tex2D(inTexture, txIndex, tyIndex);
		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				unsigned char currPx = tex2D(inTexture, txIndex + dx, tyIndex + dy);
				double w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian(centrePx - currPx, sI);
				iFiltered += w * currPx;
				wP += w;
			}
		}
		output[tyIndex * width + txIndex] = iFiltered / wP;
	}
}

void bilateralFilter(const Mat& input, Mat& output, int r, double sI, double sS)
{
	cudaEvent_t start, stop;
	float timerGPU;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int gray_size = input.step * input.rows;

	size_t pitch;                                                      
	unsigned char* d_input = NULL;
	unsigned char* d_output;

	updateGaussian(r, sS);

	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char) * input.step, input.rows); 
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char) * input.step, sizeof(unsigned char) * input.step, input.rows, cudaMemcpyHostToDevice); // create input padded with pitch
	cudaBindTexture2D(0, inTexture, d_input, input.step, input.rows, pitch); 
	cudaMalloc<unsigned char>(&d_output, gray_size); 

	dim3 block(BLOCKDIM, BLOCKDIM);

	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	cudaEventRecord(start, 0);
	gpuCalculation << <grid, block >> > (d_input, d_output, input.cols, input.rows, r, sI, sS);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);

	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventElapsedTime(&timerGPU, start, stop);
	printf("\n GPU time %f msec\n", timerGPU);
}

int main() {
	cudaEvent_t start, stop;
	float timerCPU;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	Mat src = imread("cat.bmp", IMREAD_GRAYSCALE);
	Mat dstGpu(src.rows, src.cols, CV_8UC1);
	Mat dstCpu;
	bilateralFilter(src, dstGpu, 4, 80.0, 80.0);
	cudaEventRecord(start, 0);
	cv::bilateralFilter(src, dstCpu, 9, 80, 80);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerCPU, start, stop);
	printf("\n CPU time %f msec\n", timerCPU);
	imshow("imageCPU", dstCpu);
	imshow("imageGPU", dstGpu);
	imwrite("imageCPU.bmp", dstCpu);
	imwrite("imageGPU.bmp", dstGpu);
	cv::waitKey();
}