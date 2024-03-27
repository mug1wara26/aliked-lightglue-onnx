#pragma once

#include <iostream>
#include <opencv2/core/types.hpp>

#include "Config.h"
#include "onnxruntime_cxx_api.h"

class IOnnxRunner
{
public:
    IOnnxRunner() {}
    virtual void init(const OnnxConfig& cfg) {}
    virtual void doInference(cv::Mat input) {}
};

