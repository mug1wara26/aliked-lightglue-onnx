#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Aliked/OnnxRunner.h"
#include <chrono>

using namespace cv;
using namespace std;

void showImageWithKeypoints(cv::Mat img, cv::Mat keypoints)
{
    for (int i = 0; i < keypoints.rows; i++) 
    {
        unsigned short x = keypoints.at<unsigned short>(i, 0);
        unsigned short y = keypoints.at<unsigned short>(i, 1);

        cv::circle(img, cv::Point(x, y), 1, cv::Scalar(0,255,0), -1, 8, 0);
    }

    cv::imshow("Display window", img);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cout << "Usage: ./test_aliked model_dir image_path model_type" << std::endl;
        return -1;
    }

    const string model_dir = string(argv[1]);
    const string image_path = string(argv[2]);
    const int model_type = strtol(argv[3], NULL, 10);

    cv::Mat image = imread(image_path);

    if(image.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    auto mOnnx = std::shared_ptr<AlikedRunner>(
            new AlikedRunner(model_dir, model_type)
            );

    mOnnx->inference(image);


    cv::Mat kpts = mOnnx->getProcessedKeypoints();
    showImageWithKeypoints(image.clone(), kpts);

    return 0;
}
