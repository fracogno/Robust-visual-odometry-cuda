// Copyright 2016 Robert Maier, Technical University Munich
#ifndef DVO_CUH
#define DVO_CUH

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>

#include "cublas_v2.h"

typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;


class DVO
{
public:
    enum MinimizationAlgo
    {
        GaussNewton = 0,
        GradientDescent = 1,
        LevenbergMarquardt = 2
    };

    enum WeightFunction
    {
        Huber = 0,
        Tukey = 1,
        Constant = 2
    };

    DVO();
    ~DVO();

    void init(int w, int h, const Eigen::Matrix3f &K);

    void buildPyramids(const cv::Mat &depth, const cv::Mat &gray, std::vector<float*> &depthPyramid, std::vector<float*> &grayPyramid);
    void buildPyramidCuda(const cv::Mat &imgMat, std::vector<float*> &pyramid, int mode);

    void freePyramidCuda(std::vector<float*> pyramid);

    void align(const std::vector<float*> &depthRefPyramid, const std::vector<float*> &grayRefPyramid, const std::vector<float*> &grayCurPyramid,
                    Eigen::Matrix4f &pose);

private:
    void convertSE3ToTf(const Vec6f &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t);
    void convertSE3ToTf(const Vec6f &xi, Eigen::Matrix4f &pose);
    void convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Vec6f &xi);
    void convertTfToSE3(const Eigen::Matrix4f &pose, Vec6f &xi);

    void computeGradientCuda(const float *gray, float *gradient, int direction, int w, int h);

    float calculateResidualErrorCuda(const float* d_residuals, int n);
    void calculateErrorImage(const float* residuals, int w, int h, cv::Mat &errorImage);

    void calculateMeanStdDevCuda(float* d_residuals, float &stdDev, int n);

    void computeWeightsCuda_Huber(const float* residuals, float* weights, int n, float k);
    void computeWeightsCuda_Tukey(const float* residuals, float* weights, int n, float k);
    void computeWeightsCuda_Constant(const float* residuals, float* weights, int n);

    void applyWeightsCuda(const float* weights, float* residuals, int n);

    void deriveAnalyticCuda(const float *d_grayRef, const float *d_depthRef,
                       const float *d_grayCur,
                       const float *d_gradX, const float *d_gradY,
                       const Eigen::VectorXf &xi, int lvl,
                       float* d_residuals, float* d_J, int w, int h);

    void compute_JtRCuda(const float* d_J, const float* d_residuals, Vec6f &b, int validRows);
    void compute_JtJCuda(const float* d_J, Mat6f &A, const float* d_weights, int validRows, bool useWeights);

    int numPyramidLevels_;
    std::vector<cv::Size> sizePyramid_;
    bool useWeights_;
    int numIterations_;

    // Variables for motion prior
    bool useMotionPrior_;
    Mat6f sigmaInv_ = Mat6f::Zero();
    Vec6f previousDelta_ = Vec6f::Zero();

    // these all store <numPyramidLevels_> device pointers
    std::vector<float*> d_kPyramid_;
    std::vector<float*> d_gradX_;
    std::vector<float*> d_gradY_;
    std::vector<float*> d_residuals_;
    std::vector<float*> d_weights_;
    std::vector<float*> d_J_;

    // Save these device pointers here, so we do not have to alloc and free them every time
    float *d_rotAndT_;
    float *d_rotMat_;
    float *d_t_;

    // For calculateResidualError
    float *d_squaredResiduals_;
    float *d_valids_;

    MinimizationAlgo algo_;
    WeightFunction weightfunc_;
    cublasHandle_t cHandle_;
};

#endif
