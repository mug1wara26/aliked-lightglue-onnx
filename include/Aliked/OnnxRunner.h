#pragma once

#include "Aliked/AlikedOnnxManager.h"

class AlikedRunner
{
private:
    std::unique_ptr<AlikedOnnxManager> mpOnnx;
    cv::Mat keypoints;
    cv::Mat processed_keypoints;
    cv::Mat descriptors;
    std::vector<float> scores;
public:
    OnnxConfig cfg;

    AlikedRunner(const std::string& model_path, const int model_type);

    void inference(cv::Mat img); // needs to be pass by value

    cv::Mat getKeypoints();
    cv::Mat getKeypoints(float thresh);
    cv::Mat getProcessedKeypoints();
    cv::Mat getDescriptors();
    std::vector<float> getScores();
};

