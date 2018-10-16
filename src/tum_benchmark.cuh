#ifndef TUM_BENCHMARK_H
#define TUM_BENCHMARK_H

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


bool loadAssoc(const std::string &assocFile, std::vector<std::string> &filesDepth, std::vector<std::string> &filesColor,
               std::vector<double> &timestampsDepth, std::vector<double> &timestampsColor)
{
    if (assocFile.empty())
        return false;

    //load transformations from CVPR RGBD datasets benchmark
    std::ifstream assocIn;
    assocIn.open(assocFile.c_str());
    if (!assocIn.is_open())
        return false;

    // first load all groundtruth timestamps and poses
    std::string line;
    while (std::getline(assocIn, line))
    {
        if (line.empty() || line.compare(0, 1, "#") == 0)
            continue;
        std::istringstream iss(line);
        double timestampDepth, timestampColor;
        std::string fileDepth, fileColor;
        if (!(iss >> timestampColor >> fileColor >> timestampDepth >> fileDepth))
            break;

        filesDepth.push_back(fileDepth);
        filesColor.push_back(fileColor);
        timestampsDepth.push_back(timestampDepth);
        timestampsColor.push_back(timestampColor);
    }
    assocIn.close();

    return true;
}


cv::Mat loadColor(const std::string &filename)
{
    cv::Mat imgColor = cv::imread(filename);
    // convert gray to float
    cv::Mat color;
    imgColor.convertTo(color, CV_32FC3, 1.0f / 255.0f);
    return color;
}


cv::Mat loadGray(const std::string &filename)
{
    cv::Mat imgGray = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    // convert gray to float
    cv::Mat gray;
    imgGray.convertTo(gray, CV_32FC1, 1.0f / 255.0f);
    return gray;
}


cv::Mat loadDepth(const std::string &filename)
{
    //fill/read 16 bit depth image
    cv::Mat imgDepthIn = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    cv::Mat imgDepth;
    imgDepthIn.convertTo(imgDepth, CV_32FC1, (1.0 / 5000.0));
    return imgDepth;
}


bool savePoses(const std::string &filename, const std::vector<Eigen::Matrix4f> &poses, const std::vector<double> &timestamps)
{
    if (filename.empty())
        return false;

    // open output file for TUM RGB-D benchmark poses
    std::ofstream outFile;
    outFile.open(filename.c_str());
    if (!outFile.is_open())
        return false;

    // write poses into output file
    outFile << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < poses.size(); i++)
    {
        Eigen::Matrix4f pose = poses[i];

        //write into evaluation file
        //timestamp
        double timestamp = timestamps[i];
        outFile << timestamp << " ";

        //translation
        Eigen::Vector3f translation = pose.topRightCorner(3,1);
        outFile << translation[0] << " " << translation[1] << " " << translation[2];
        //rotation (quaternion)
        Eigen::Matrix3f rot = pose.topLeftCorner(3,3);
        Eigen::Quaternionf quat(rot);
        outFile << " " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
    }
    outFile.close();

    return true;
}

#endif
