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

#include <OpenNI.h>
#include <PS1080.h>

#include "dvo.cuh"
#include "tum_benchmark.cuh"

#define STR1(x)  #x
#define STR(x)  STR1(x)

void generateOffFile(std::string dataFolder, std::string rgbImage, std::string depthImage, Eigen::Matrix3f intrinsics, std::vector<Eigen::Matrix4f> poses);


int main(int argc, char *argv[])
{
    openni::Device device;
    openni::VideoStream colorStream, depthStream;
    openni::VideoFrameRef colorFrame, depthFrame;

    openni::VideoMode cameraModeDepth; 
    cameraModeDepth.setResolution(320, 240);
    cameraModeDepth.setFps(30);

    // Initialize OpenNI video capture
    openni::OpenNI::initialize();
    device.open(openni::ANY_DEVICE);

    // Create depth camera stream
    depthStream.create(device, openni::SENSOR_DEPTH);
    depthStream.setVideoMode(cameraModeDepth);
    depthStream.start();

    // Create color camera stream
    colorStream.create(device, openni::SENSOR_COLOR);
    colorStream.start();

    // Set flag for synchronization between color camera and depth camera
    // Set flag for registration between color camera and depth camera
    device.setDepthColorSyncEnabled(true);
    device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

    // Get focal length of IR camera in mm for VGA resolution
    double pixelSize = 0;
    depthStream.getProperty<double>(XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE, &pixelSize);
    
    int zeroPlaneDistance = 0; //focal in mm
    depthStream.getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE, &zeroPlaneDistance);
    float depthFocalLength_VGA = (float) zeroPlaneDistance / pixelSize;
    std::cout << depthFocalLength_VGA << std::endl;

    Eigen::Matrix3f K;
    /*K <<    depthFocalLength_VGA, 0.0, 159.5,
            0.0, depthFocalLength_VGA, 219.5,
            0.0, 0.0, 1.0;*/
    K <<    517.5, 0.0, 159.5,
            0.0, 517.5, 219.5,
            0.0, 0.0, 1.0;

    Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Matrix4f> poses;
    poses.push_back(absPose);
    std::vector<double> timestamps;
    timestamps.push_back(0);

    int nColsDepth = depthStream.getVideoMode().getResolutionX();
    int nRowsDepth = depthStream.getVideoMode().getResolutionY();

    int nColsColor = colorStream.getVideoMode().getResolutionX();
    int nRowsColor = colorStream.getVideoMode().getResolutionY();

    int w = nColsDepth;
    int h = nRowsDepth;

    cv::Mat grayRef;
    cv::Mat depthRef;
    cv::Mat tmp, tmp2;

    depthStream.readFrame(&depthFrame);
    if(depthFrame.isValid()){
	tmp = cv::Mat(nRowsDepth, nColsDepth, CV_16UC1, (char*)depthFrame.getData());
    	cv::flip(tmp, tmp2, 1);        
	tmp2.convertTo(depthRef, CV_32FC1, (1.0 / 5000.0));
    }
    else
	std::cout << "Error 1" << std::endl;

    colorStream.readFrame(&colorFrame);
    if(colorFrame.isValid()){
	tmp = cv::Mat(nRowsColor, nColsColor, CV_8UC3, (char*)colorFrame.getData());
	cv::flip(tmp, tmp2, 1); 
	cv::cvtColor(tmp2, tmp2, CV_BGR2GRAY);
	tmp2.convertTo(grayRef, CV_32FC1, 1.0f / 255.0f);
    }
    else
	std::cout << "Error 2" << std::endl;

    DVO dvo;
    dvo.init(w, h, K);

    std::vector<float*> grayRefPyramid; // this will store device pointers
    std::vector<float*> depthRefPyramid; // this will store device pointers
    dvo.buildPyramids(depthRef, grayRef, depthRefPyramid, grayRefPyramid);

    std::ofstream outFile("../data/trajectory_points.txt");
    if (!outFile.is_open()) return;

    // process frames
    double runtimeAvg = 0.0;
    int framesProcessed = 0;
    while (true)
    {       
	cv::Mat grayCur;
        cv::Mat depthCur;

        depthStream.readFrame(&depthFrame);
        if(depthFrame.isValid()){
	    tmp = cv::Mat(nRowsDepth, nColsDepth, CV_16UC1, (char*)depthFrame.getData()); 
	    cv::flip(tmp, tmp2, 1);        
	    tmp2.convertTo(depthCur, CV_32FC1, (1.0 / 5000.0));
	}
	else
	    std::cout << "Error 1" << std::endl;

        colorStream.readFrame(&colorFrame);
        if(colorFrame.isValid()){
	    tmp = cv::Mat(nRowsColor, nColsColor, CV_8UC3, (char*)colorFrame.getData());
	    cv::flip(tmp, tmp2, 1); 
	    cv::cvtColor(tmp2, tmp2, CV_BGR2GRAY);
	    tmp2.convertTo(grayCur, CV_32FC1, 1.0f / 255.0f);
	}      
	else
	    std::cout << "Error 2" << std::endl;  

	// Ignore first second to set things up
	if(framesProcessed < 30){
		framesProcessed++;
		continue;
	}
	//std::cout << "aligning frames " << (framesProcessed-30) << " and " << (framesProcessed+1-30) << std::endl;
	
        // build pyramid
        std::vector<float*> grayCurPyramid; // this will store device pointers
        std::vector<float*> depthCurPyramid; // this will store device pointers
        dvo.buildPyramids(depthCur, grayCur, depthCurPyramid, grayCurPyramid);

        // Show images
        cv::imshow("colorImage", grayCur);
        cv::imshow("depthImage", depthCur);

	assert(grayCur.rows == depthCur.rows);
	assert(grayCur.cols == depthCur.cols);

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

	Eigen::Matrix3f tmprot = absPose.block(0, 0, 3, 3);
        Eigen::Vector3f tmptra = absPose.block(0, 3, 3, 1);
        Eigen::Vector3f camPos = - tmprot.transpose() * tmptra;
        outFile << -camPos.x() << " " << camPos.y() << std::endl;

	// free ref pyramids
        dvo.freePyramidCuda(depthRefPyramid);
        dvo.freePyramidCuda(grayRefPyramid);

	depthRefPyramid = depthCurPyramid;
        grayRefPyramid = grayCurPyramid;	
	timestamps.push_back(0);

	++framesProcessed;
        int key = cv::waitKey(5);
	if(key == 113) break; // Press Q and quit
    }

    std::cout << "average runtime: " << (runtimeAvg / (framesProcessed-30)) * 1000.0 << " ms" << std::endl;

    // save poses
    savePoses("../data/traj.txt", poses, timestamps);

    // clean up
    dvo.freePyramidCuda(depthRefPyramid);
    dvo.freePyramidCuda(grayRefPyramid);

    // destroy image streams and close the OpenNI device
    colorStream.destroy(); 
    depthStream.destroy();
    device.close(); 
    openni::OpenNI::shutdown(); 

    cv::destroyAllWindows();
    std::cout << "Direct Image Alignment finished." << std::endl;


    //generateOffFile(dataFolder, filesColor[0], filesDepth[0], K, poses);

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

