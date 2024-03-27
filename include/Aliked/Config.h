#pragma once

#include <string>

struct OnnxConfig
{
    std::string mEnginePath;
    std::string device;
    std::string modelName;

    int width, height, dims, channels, k;
    bool nchw = false;
};

