#pragma once

#include "Lightglue/LightglueOnnxManager.h"

class LightglueRunner
{
private:
    std::unique_ptr<LightglueOnnxManager> mpOnnx;
    cv::Mat matches;
    std::vector<float> scores;
public:
    LightglueConf cfg;

    LightglueRunner(const std::string& model_path, const int extractor_type);

    void inference(cv::Mat kpt0, cv::Mat kpt1, cv::Mat desc0, cv::Mat desc1); // needs to be pass by value
    cv::Mat getMatches();
    std::vector<float> getScores();
};

