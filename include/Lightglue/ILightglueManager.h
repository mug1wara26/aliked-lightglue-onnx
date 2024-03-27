#pragma once

#include <iostream>
#include <opencv2/core/types.hpp>

#include "LightglueConf.h"
#include "onnxruntime_cxx_api.h"

class ILightglueManager
{
public:
    ILightglueManager() {}
    virtual void init(const LightglueConf& cfg) {}
    virtual std::vector<Ort::Value> doInference(cv::Mat kpts0, cv::Mat kpts1, cv::Mat desc0, cv::Mat desc1) {}
};

