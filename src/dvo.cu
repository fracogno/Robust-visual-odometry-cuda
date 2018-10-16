// Copyright 2016 Robert Maier, Technical University Munich
#include "dvo.cuh"
#include "helper.cuh"
#include "dvo_cuda_kernels.cuh"
#include "cublas_v2.h"

#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Cholesky>
#include <sophus/se3.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>


DVO::DVO() :
    numPyramidLevels_(5),
    useWeights_(true),
    useMotionPrior_(false),
    numIterations_(100),
    algo_(GaussNewton),
    weightfunc_(Huber),
    d_kPyramid_(numPyramidLevels_),
    d_gradX_(numPyramidLevels_),
    d_gradY_(numPyramidLevels_),
    d_residuals_(numPyramidLevels_),
    d_weights_(numPyramidLevels_),
    d_J_(numPyramidLevels_)
{
}


DVO::~DVO()
{
    cublasDestroy_v2(cHandle_);

    cudaFree(d_rotMat_); CUDA_CHECK;
    cudaFree(d_t_); CUDA_CHECK;
    cudaFree(d_rotAndT_); CUDA_CHECK;

    cudaFree(d_squaredResiduals_); CUDA_CHECK;
    cudaFree(d_valids_); CUDA_CHECK;

    for (int i = 0; i < numPyramidLevels_; ++i)
    {
        cudaFree(d_kPyramid_[i]); CUDA_CHECK;
        cudaFree(d_gradX_[i]); CUDA_CHECK;
        cudaFree(d_gradY_[i]); CUDA_CHECK;
        cudaFree(d_residuals_[i]); CUDA_CHECK;
        cudaFree(d_weights_[i]); CUDA_CHECK;
        cudaFree(d_J_[i]); CUDA_CHECK;
    }
}


void DVO::init(int w, int h, const Eigen::Matrix3f &K)
{
    // init cuda context
    cudaDeviceSynchronize();
    cublasCreate_v2(&cHandle_);

    // pyramid level size
    int wDown = w;
    int hDown = h;
    int n = wDown*hDown;
    sizePyramid_.push_back(cv::Size(wDown, hDown));

    cudaMalloc(&(d_gradX_[0]), n*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&(d_gradY_[0]), n*sizeof(float)); CUDA_CHECK;

    // Jacobian
    cudaMalloc(&(d_J_[0]), n*6*sizeof(float)); CUDA_CHECK;

    // residuals
    cudaMalloc(&(d_residuals_[0]), n*sizeof(float)); CUDA_CHECK;

    // per-residual weights
    cudaMalloc(&(d_weights_[0]), n*sizeof(float)); CUDA_CHECK;

    // camera matrix
    cudaMalloc(&(d_kPyramid_[0]), 9*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(d_kPyramid_[0], K.data(), 9*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    Eigen::Matrix3f currentK = K;

    // Motion prior
    if(useMotionPrior_) {
        sigmaInv_(0,0) = 1.f / 0.00001f;
        sigmaInv_(1,1) = 1.f / 0.00001f;
        sigmaInv_(2,2) = 1.f / 0.00001f;
        sigmaInv_(3,3) = 1.f / 0.00001f;
        sigmaInv_(4,4) = 1.f / 0.00001f;
        sigmaInv_(5,5) = 1.f / 0.00001f;
        std::cout << "SigmaInv =\n" << sigmaInv_ << std::endl;
    }

    cudaMalloc(&d_rotMat_, 9*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_t_, 3*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_rotAndT_, 12*sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_squaredResiduals_, n*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_valids_, n*sizeof(float)); CUDA_CHECK;

    for (int i = 1; i < numPyramidLevels_; ++i)
    {
        // pyramid level size
        wDown = wDown / 2;
        hDown = hDown / 2;
        int n = wDown*hDown;
        sizePyramid_.push_back(cv::Size(wDown, hDown));

        cudaMalloc(&(d_gradX_[i]), wDown*hDown*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&(d_gradY_[i]), wDown*hDown*sizeof(float)); CUDA_CHECK;

        // Jacobian
        cudaMalloc(&(d_J_[i]), n*6*sizeof(float)); CUDA_CHECK;

        // residuals
        cudaMalloc(&(d_residuals_[i]), n*sizeof(float)); CUDA_CHECK;

        // per-residual weights
        cudaMalloc(&(d_weights_[i]), n*sizeof(float)); CUDA_CHECK;

        // downsample camera matrix
        Eigen::Matrix3f kDown = currentK;
        kDown(0, 2) += 0.5f;
        kDown(1, 2) += 0.5f;
        kDown.topLeftCorner(2, 3) = kDown.topLeftCorner(2, 3) * 0.5f;
        kDown(0, 2) -= 0.5f;
        kDown(1, 2) -= 0.5f;
        currentK = kDown;

        cudaMalloc(&(d_kPyramid_[i]), 9*sizeof(float)); CUDA_CHECK;
        cudaMemcpy(d_kPyramid_[i], kDown.data(), 9*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *          S E 3   A N D   T F   C O N V E R T   F U N C T I O N S
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void DVO::convertSE3ToTf(const Vec6f &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t)
{
    // rotation
    Sophus::SE3f se3 = Sophus::SE3f::exp(xi);
    Eigen::Matrix4f mat = se3.matrix();
    rot = mat.topLeftCorner(3, 3);
    t = mat.topRightCorner(3, 1);
}


void DVO::convertSE3ToTf(const Vec6f &xi, Eigen::Matrix4f &pose)
{
    Sophus::SE3f se3 = Sophus::SE3f::exp(xi);
    pose = se3.matrix();
}


void DVO::convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Vec6f &xi)
{
    Sophus::SE3f se3(rot, t);
    xi = Sophus::SE3f::log(se3);
}


void DVO::convertTfToSE3(const Eigen::Matrix4f &pose, Vec6f &xi)
{
    Eigen::Matrix3f rot = pose.topLeftCorner(3, 3);
    Eigen::Vector3f t = pose.topRightCorner(3, 1);
    convertTfToSE3(rot, t, xi);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *          C U D A   F U N C T I O N S
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void DVO::computeGradientCuda(const float *d_gray, float *d_gradient, int direction, int w, int h)
{
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid2D(block, w, h);

    computeGradientKernel<<<grid, block>>>(d_gray, d_gradient, direction, w, h);

    CUDA_CHECK;
}


float DVO::calculateResidualErrorCuda(const float* d_residuals, int n)
{
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid1D(block, n);
    getSquaredResidualsKernel<<<grid, block>>>(d_residuals, d_squaredResiduals_, d_valids_, n);
    CUDA_CHECK;

    float error = 0.f;
    cublasSasum_v2(cHandle_, n, d_squaredResiduals_, 1, &error);
    float numValids = 0.f;
    cublasSasum_v2(cHandle_, n, d_valids_, 1, &numValids);

    if(numValids > 0.f) {
        error = error / static_cast<float>(numValids);
    }

    return error;
}


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const {
            return x * x;
        }
};


void DVO::calculateMeanStdDevCuda(float* d_residuals, float &stdDev, int n)
{
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_residuals);
    thrust::device_vector<float> vec(dev_ptr, dev_ptr + n);
    float sum = thrust::reduce(vec.begin(), vec.end(), (float) 0.f, thrust::plus<float>());

    float thrustMean = (sum / (float)n);

    thrust::device_vector<float> diffRes(n);
    thrust::transform(vec.begin(), vec.end(),
                      thrust::make_constant_iterator(thrustMean),
                      diffRes.begin(),
                      thrust::minus<float>());

    square<float>        unary_op;
    thrust::plus<float> binary_op;
    float init = 0.f;

    float variance = thrust::transform_reduce(diffRes.begin(), diffRes.end(), unary_op, init, binary_op);
    stdDev = std::sqrt(variance / (float)n);
}


void DVO::computeWeightsCuda_Huber(const float* residuals, float* weights, int n, float k)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid1D(block, n);
    
    // run cuda kernel
    computeWeightsKernel_Huber<<<grid,block>>>(residuals, weights, n, k);

    // check for errors
    CUDA_CHECK;
}

void DVO::computeWeightsCuda_Tukey(const float* residuals, float* weights, int n, float k)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid1D(block, n);

    // run cuda kernel
    computeWeightsKernel_Tukey<<<grid,block>>>(residuals, weights, n, k);

    // check for errors
    CUDA_CHECK;
}

void DVO::computeWeightsCuda_Constant(const float* residuals, float* weights, int n)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid1D(block, n);

    // run cuda kernel
    computeWeightsKernel_Constant<<<grid,block>>>(residuals, weights, n);

    // check for errors
    CUDA_CHECK;
}


void DVO::applyWeightsCuda(const float* weights, float* residuals, int n)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid1D(block, n);
    
    // run cuda kernel
    applyWeightsKernel<<<grid,block>>>(weights, residuals, n);

    // check for errors
    CUDA_CHECK;
}


void DVO::compute_JtRCuda(const float* d_J, const float* d_residuals, Vec6f &b, int validRows)
{
    int n = 6;
    int m = validRows;

    float *d_b;
    cudaMalloc(&d_b, n*sizeof(float)); CUDA_CHECK;

    cudaMemset(d_b, 0, n*sizeof(float)); CUDA_CHECK;

    dim3 block(32, 8, 1);
    dim3 grid = computeGrid2D(block, n, m);

    compute_JtR_Kernel<<<grid, block>>>(d_J, d_residuals, d_b, m, n);
    CUDA_CHECK;

    cudaMemcpy(b.data(), d_b, n*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(d_b);
}


void DVO::compute_JtJCuda(const float* d_J, Mat6f &A, const float* d_weights, int validRows, bool useWeights)
{
    int n = 6;

    float *d_A;
    cudaMalloc(&d_A, n * n * sizeof(float)); CUDA_CHECK;

    dim3 block(32, 8, 1);
    dim3 grid = computeGrid2D(block, n, n);

    compute_JTJ_Kernel<<<grid, block>>>(d_J, d_A, d_weights, validRows, useWeights); CUDA_CHECK;

    cudaMemcpy(A.data(), d_A, n * n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    cudaFree (d_A); // free device memory
}


void DVO::deriveAnalyticCuda(const float *d_grayRef, const float *d_depthRef,
                             const float *d_grayCur,
                             const float *d_gradX, const float *d_gradY,
                             const Eigen::VectorXf &xi, int lvl,
                             float* d_residuals, float* d_J, int w, int h)
{
    // copy camera instrinsics to float*
    float *d_K = d_kPyramid_[lvl];

    // convert SE3 to rotation matrix and translation vector
    Eigen::Matrix3f rotMat;
    Eigen::Vector3f t;
    convertSE3ToTf(xi, rotMat, t);

    // copy rotation matrix and translation vector to device
    cudaMemcpy(d_rotMat_, rotMat.data(), 9*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_t_, t.data(), 3*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    // // calculate per-pixel residuals
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid2D(block, w, h);
    calculateErrorKernel<<<grid, block>>>(d_grayRef, d_depthRef, d_grayCur, d_rotMat_, d_t_, d_K, d_residuals, w, h);
    CUDA_CHECK;

    // create and fill Jacobian
    deriveAnalyticKernel<<<grid, block>>>(d_depthRef, d_rotMat_, d_t_, d_K, d_gradX, d_gradY, d_J, w, h);
    CUDA_CHECK;
}


/**
 * @brief DVO::buildPyramidCuda creates the resolution pyramid for the given cv::Mat.
 * The image data in the pyramid is saved as device pointers.
 * @param imgMat
 * @param pyramid
 * @param mode 0 for gray, 1 for depth
 */
void DVO::buildPyramidCuda(const cv::Mat &imgMat, std::vector<float*> &pyramid, int mode)
{
    // Dimensions for image
    int w = imgMat.cols;
    int h = imgMat.rows;
    int wDown = w;
    int hDown = h;

    // Host pointer
    float *img = new float[w * h];

    // Push original size images on pyramid
    convertMatToLayered(img, imgMat);

    // Device pointer
    float *d_img;
    cudaMalloc(&d_img, w * h* sizeof(float)); CUDA_CHECK;

    // Convert original image to layered and copy to device
    convertMatToLayered(img, imgMat);
    cudaMemcpy(d_img, img, w * h* sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    pyramid.push_back(d_img);

    // Block and grid dimenions
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid2D(block, w, h);

    for (int i = 1; i < numPyramidLevels_; ++i)
    {
        wDown /= 2;
        hDown /= 2;

        float *d_downImg;
        cudaMalloc(&d_downImg, wDown * hDown * sizeof(float)); CUDA_CHECK;
        cudaMemset(d_downImg, 0.f, wDown * hDown); CUDA_CHECK;

        // Call Cuda kernel
        if(mode == 0)
        {
            downsampleGreyKernel<<<grid, block>>>(d_downImg, d_img, wDown*2, hDown*2); CUDA_CHECK;
        }
        else
        {
            downsampleDepthKernel<<<grid, block>>>(d_downImg, d_img, wDown*2, hDown*2); CUDA_CHECK;
        }

        pyramid.push_back(d_downImg);
        d_img = d_downImg; // for next level
    }

    delete[] img;
}

void DVO::freePyramidCuda(std::vector<float*> pyramid) {
    for(int i = 0; i < numPyramidLevels_; i++) {
        cudaFree(pyramid[i]); CUDA_CHECK;
    }
}


void DVO::buildPyramids(const cv::Mat &depth, const cv::Mat &gray, std::vector<float*> &depthPyramid, std::vector<float*> &grayPyramid)
{
    buildPyramidCuda(gray, grayPyramid, 0);
    buildPyramidCuda(depth, depthPyramid, 1);
}


void DVO::calculateErrorImage(const float* residuals, int w, int h, cv::Mat &errorImage)
{
    cv::Mat imgResiduals = cv::Mat::zeros(h, w, CV_32FC1);
    float* ptrResiduals = (float*)imgResiduals.data;

    // fill residuals image
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            int off = y*w + x;
            if (residuals[off] != 0.0f)
                ptrResiduals[off] = residuals[off];
        }
    }

    imgResiduals.convertTo(errorImage, CV_8SC1, 127.0);
}


void DVO::align(const std::vector<float*> &depthRefPyramid, const std::vector<float*> &grayRefPyramid,
                const std::vector<float*> &grayCurPyramid,
                Eigen::Matrix4f &pose)
{
    Vec6f xi;
    convertTfToSE3(pose, xi);

    Vec6f lastXi = Vec6f::Zero();

    int maxLevel = numPyramidLevels_-1;
    int minLevel = 1;
    float initGradDescStepSize = 1e-3f;
    float gradDescStepSize = initGradDescStepSize;

    Mat6f A;
    Mat6f diagMatA = Mat6f::Identity();
    Vec6f delta = Vec6f::Zero();

    for (int lvl = maxLevel; lvl >= minLevel; --lvl)
    {
        float lambda = 0.1f;

        int w = sizePyramid_[lvl].width;
        int h = sizePyramid_[lvl].height;
        int n = w*h;

        float *d_grayRef = grayRefPyramid[lvl];
        float *d_depthRef = depthRefPyramid[lvl];
        float *d_grayCur = grayCurPyramid[lvl];

        // compute gradient images
        computeGradientCuda(d_grayCur, d_gradX_[lvl], 0, w, h);
        computeGradientCuda(d_grayCur, d_gradY_[lvl], 1, w, h);

        float errorLast = std::numeric_limits<float>::max();
        for (int itr = 0; itr < numIterations_; ++itr) {
            // compute residuals and Jacobian
            deriveAnalyticCuda(d_grayRef, d_depthRef, d_grayCur, d_gradX_[lvl], d_gradY_[lvl], xi, lvl, d_residuals_[lvl], d_J_[lvl], w, h);

            if (useWeights_) {
                // compute mean and standard deviation
                float stdDev;
                calculateMeanStdDevCuda(d_residuals_[lvl], stdDev, n);

                if (weightfunc_ == Huber) {
                    // compute robust Huber weights
                    float k = 1.345f * stdDev;
                    // compute robust weights
                    computeWeightsCuda_Huber(d_residuals_[lvl], d_weights_[lvl], n, k);
                } else if (weightfunc_ == Tukey) {
                    // http://users.stat.umn.edu/~sandy/courses/8053/handouts/robust.pdf
                    // compute robust Tukey weights
                    float k = 4.685f * stdDev;
                    // compute robust weights
                    computeWeightsCuda_Tukey(d_residuals_[lvl], d_weights_[lvl], n, k);
                } else if (weightfunc_ == Constant){
                    //Constant weights
                    computeWeightsCuda_Constant(d_residuals_[lvl], d_weights_[lvl], n);
                }
                // apply robust weights
                applyWeightsCuda(d_weights_[lvl], d_residuals_[lvl], n);
            }

            // compute update
            Vec6f b;
            compute_JtRCuda(d_J_[lvl], d_residuals_[lvl], b, n);

            if (algo_ == GradientDescent) {
                // Gradient Descent
                delta = -gradDescStepSize * b * (1.0f / b.norm());
            } else if (algo_ == GaussNewton) {
                // Gauss-Newton algorithm
                compute_JtJCuda(d_J_[lvl], A, d_weights_[lvl], n, useWeights_);

                if(useMotionPrior_) {
                    A += sigmaInv_;

                    Vec6f motionDiff = previousDelta_ - delta;
                    b += sigmaInv_ * motionDiff;
                }

                // solve using Cholesky LDLT decomposition
                delta = (A.ldlt().solve(-b));
            } else if (algo_ == LevenbergMarquardt) {
                // Levenberg-Marquardt algorithm
                compute_JtJCuda(d_J_[lvl], A, d_weights_[lvl], n, useWeights_);
                diagMatA.diagonal() = lambda * A.diagonal();
                delta = -((A + diagMatA).ldlt().solve(b));
            }

            // apply update: left-multiplicative increment on SE3
            lastXi = xi;
            xi = Sophus::SE3f::log(Sophus::SE3f::exp(delta) * Sophus::SE3f::exp(xi));

            // compute error
            float error = calculateResidualErrorCuda(d_residuals_[lvl], n);

            if (algo_ == LevenbergMarquardt) {
                if (error >= errorLast) {
                    lambda = lambda * 5.0f;
                    xi = lastXi;

                    if (lambda > 5.0f)
                        break;
                } else {
                    lambda = lambda / 1.5f;
                }
            } else if (algo_ == GaussNewton) {
                // break if no improvement (0.99 or 0.995)
                if (error / errorLast > 0.995f) {
                    break;
                }
            } else if (algo_ == GradientDescent) {
                if (error >= errorLast) {
                    gradDescStepSize = gradDescStepSize * 0.5f;
                    if (gradDescStepSize <= initGradDescStepSize * 0.01f)
                        gradDescStepSize = initGradDescStepSize * 0.01f;
                    xi = lastXi;
                } else {
                    gradDescStepSize = gradDescStepSize * 2.0f;
                    if (gradDescStepSize >= initGradDescStepSize * 100.0f)
                        gradDescStepSize = initGradDescStepSize * 100.0f;

                    // break if no improvement (0.99 or 0.995)
                    if (error / errorLast > 0.995f)
                        break;
                }
            }

            errorLast = error;
        }
    }

    previousDelta_ = delta;

    // store to output pose
    convertSE3ToTf(xi, pose);
}
