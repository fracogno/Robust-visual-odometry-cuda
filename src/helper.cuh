// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_HELPER_CUH
#define TUM_HELPER_CUH

#include <cuda_runtime.h>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>


// CUDA utility functions

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

// compute grid size from block size
inline dim3 computeGrid1D(const dim3 &block, const int w)
{
    return dim3((block.x + w - 1) / block.x, 1, 1);
}

inline dim3 computeGrid2D(const dim3 &block, const int w, const int h)
{
    return dim3((block.x + w - 1) / block.x, (h + block.y - 1) / block.y, 1);
}

inline dim3 computeGrid3D(const dim3 &block, const int w, const int h, const int s)
{
    return dim3(0, 0, 0);   // TODO (3.2) compute 3D grid size from block size
}


// OpenCV image conversion
// interleaved to layered
void convertMatToLayered(float *aOut, const cv::Mat &mIn);

// layered to interleaved
void convertLayeredToMat(cv::Mat &mOut, const float *aIn);


// OpenCV GUI functions
// open camera
bool openCamera(cv::VideoCapture &camera, int device, int w = 640, int h = 480);

// show image
void showImage(std::string title, const cv::Mat &mat, int x, int y);

// show histogram
void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY);


// adding Gaussian noise
void addNoise(cv::Mat &m, float sigma);


// measuring time
class Timer
{
public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
        cudaDeviceSynchronize();
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
private:
	clock_t tStart;
	bool running;
	float sec;
};

#endif
