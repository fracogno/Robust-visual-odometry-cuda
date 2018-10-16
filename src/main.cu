#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "dvo.cuh"
#include "tum_benchmark.cuh"

#define STR1(x)  #x
#define STR(x)  STR1(x)

void generateOffFile(std::string dataFolder, std::string rgbImage, std::string depthImage, Eigen::Matrix3f intrinsics, std::vector<Eigen::Matrix4f> poses);


int main(int argc, char *argv[])
{
    std::string dataFolder = std::string(STR(DVO_SOURCE_DIR)) + "/data/rgbd_dataset_freiburg1_xyz/";

    Eigen::Matrix3f K;
#if 1
    // initialize intrinsic matrix: fr1
    K <<    517.3, 0.0, 318.6,
            0.0, 516.5, 255.3,
            0.0, 0.0, 1.0;
    //dataFolder = "/work/maierr/rgbd_data/rgbd_dataset_freiburg1_xyz/";
    //dataFolder = "/work/maierr/rgbd_data/rgbd_dataset_freiburg1_desk2/";
#else
    dataFolder = "/work/maierr/rgbd_data/rgbd_dataset_freiburg3_long_office_household/";
    // initialize intrinsic matrix: fr3
    K <<    535.4, 0.0, 320.1,
            0.0, 539.2, 247.6,
            0.0, 0.0, 1.0;
#endif
    //std::cout << "Camera matrix: " << K << std::endl;

    // load file names
    std::string assocFile = dataFolder + "assoc.txt";
    std::vector<std::string> filesColor;
    std::vector<std::string> filesDepth;
    std::vector<double> timestampsDepth;
    std::vector<double> timestampsColor;
    if (!loadAssoc(assocFile, filesDepth, filesColor, timestampsDepth, timestampsColor))
    {
        std::cout << "Assoc file could not be loaded!" << std::endl;
        return 1;
    }
    int numFrames = filesDepth.size();

    int maxFrames = -1;
    maxFrames = 400;

    Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
#if 0
    // initiate with the first pose from the 'rgbd_dataset_freiburg1_xyz' dataset
    absPose <<  0.0698161f,  0.4672371f, -0.8813712f, 1.3563f,
                0.9951546f,  0.0286956f,  0.0940415f, 0.6305f,
                0.0692311f, -0.8836663f, -0.4629698f, 1.6380f,
                0.0f,        0.0f,        0.0f,       1.0f;
#endif

    std::vector<Eigen::Matrix4f> poses;
    std::vector<double> timestamps;
    poses.push_back(absPose);
    timestamps.push_back(timestampsDepth[0]);

    cv::Mat grayRef = loadGray(dataFolder + filesColor[0]);
    cv::Mat depthRef = loadDepth(dataFolder + filesDepth[0]);
    int w = depthRef.cols;
    int h = depthRef.rows;

    DVO dvo;
    dvo.init(w, h, K);

    std::vector<float*> grayRefPyramid; // this will store device pointers
    std::vector<float*> depthRefPyramid; // this will store device pointers
    dvo.buildPyramids(depthRef, grayRef, depthRefPyramid, grayRefPyramid);

    // process frames
    double runtimeAvg = 0.0;
    int framesProcessed = 0;
    for (size_t i = 1; i < numFrames && (maxFrames < 0 || i < maxFrames); ++i)
    {
        std::cout << "aligning frames " << (i-1) << " and " << i  << std::endl;

        // load input frame
        std::string fileColor1 = filesColor[i];
        std::string fileDepth1 = filesDepth[i];
        double timeDepth1 = timestampsDepth[i];
        cv::Mat grayCur = loadGray(dataFolder + fileColor1);
        cv::Mat depthCur = loadDepth(dataFolder + fileDepth1);
        // build pyramid
        std::vector<float*> grayCurPyramid; // this will store device pointers
        std::vector<float*> depthCurPyramid; // this will store device pointers
        dvo.buildPyramids(depthCur, grayCur, depthCurPyramid, grayCurPyramid);

        // frame alignment
        double tmr = (double)cv::getTickCount();

        Eigen::Matrix4f relPose = Eigen::Matrix4f::Identity();
        dvo.align(depthRefPyramid, grayRefPyramid, grayCurPyramid, relPose);

        tmr = ((double)cv::getTickCount() - tmr)/cv::getTickFrequency();
        runtimeAvg += tmr;

        // concatenate poses
        Eigen::Matrix4f rPo = relPose.inverse();
        absPose = absPose * rPo;
        poses.push_back(absPose);
        timestamps.push_back(timeDepth1);

        // free ref pyramids
        dvo.freePyramidCuda(depthRefPyramid);
        dvo.freePyramidCuda(grayRefPyramid);

        depthRefPyramid = depthCurPyramid;
        grayRefPyramid = grayCurPyramid;
        ++framesProcessed;
    }
    std::cout << "average runtime: " << (runtimeAvg / framesProcessed) * 1000.0 << " ms" << std::endl;

    // save poses
    savePoses(dataFolder + "traj.txt", poses, timestamps);

    // clean up
    dvo.freePyramidCuda(depthRefPyramid);
    dvo.freePyramidCuda(grayRefPyramid);

    generateOffFile(dataFolder, filesColor[0], filesDepth[0], K, poses);

    cv::destroyAllWindows();
    std::cout << "Direct Image Alignment finished." << std::endl;
    return 0;
}





void generateOffFile(std::string dataFolder, std::string rgbImage, std::string depthImage, Eigen::Matrix3f intrinsics, std::vector<Eigen::Matrix4f> poses){

	std::string visualizeFilename = "visualize.off";
	std::ofstream outFile((dataFolder + visualizeFilename).c_str());
    if (!outFile.is_open()) return;

    // Load images
    cv::Mat imgColor = cv::imread(dataFolder + rgbImage, CV_LOAD_IMAGE_ANYCOLOR);
    cv::Mat imgDepthIn = cv::imread(dataFolder + depthImage, CV_LOAD_IMAGE_ANYDEPTH);

    // write header
    outFile << "COFF" << std::endl;
    outFile << (poses.size() + (imgColor.rows * imgColor.cols)) << " 0 0" << std::endl;

    // Save camera position
    for (int i = 0; i < poses.size(); ++i) {
        Eigen::Matrix4f cameraExtrinsics = poses[i];
        Eigen::Matrix3f rotation = cameraExtrinsics.block(0, 0, 3, 3);
        Eigen::Vector3f translation = cameraExtrinsics.block(0, 3, 3, 1);
        Eigen::Vector3f cameraPosition = - rotation.transpose() * translation;
        outFile << cameraPosition.x() << " " << cameraPosition.y() << " " << cameraPosition.z() << " 255 0 0" << std::endl;
    }    

    float fovX = intrinsics(0, 0);
    float fovY = intrinsics(1, 1);
    float cX = intrinsics(0, 2);
    float cY = intrinsics(1, 2);

    Eigen::Matrix4f cameraExtrinsicsInv = poses[0].inverse();
    Eigen::Matrix3f rotationInv = cameraExtrinsicsInv.block(0, 0, 3, 3);
    Eigen::Vector3f translationInv = cameraExtrinsicsInv.block(0, 3, 3, 1);

    assert(imgColor.rows == imgDepthIn.rows);
    assert(imgColor.cols == imgDepthIn.cols);

    for(int v=0; v<imgColor.rows; ++v){
    	for(int u=0;u<imgColor.cols; ++u){

    		cv::Point point2d = cv::Point(u,v);

    		float x = ((float) u - cX) / fovX;
		    float y = ((float) v - cY) / fovY;

		    float depth = imgDepthIn.at<uint16_t>(point2d);

		    depth *= 1.0 / 5000.0;

		    Eigen::Vector4f backprojected = Eigen::Vector4f(depth * x, depth * y, depth, 1);
		    Eigen::Vector4f worldSpace = cameraExtrinsicsInv * backprojected;

		    cv::Vec3b colors = imgColor.at<cv::Vec3b>(v,u);

		    outFile << worldSpace[0] << " " << worldSpace[1] << " " << worldSpace[2] << " " << 
            static_cast<unsigned>(colors[2]) << " " << static_cast<unsigned>(colors[1]) << " " << static_cast<unsigned>(colors[0]) <<  std::endl;
    	}
    }
   
}

