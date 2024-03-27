#pragma once

#include "Aliked/OnnxRunner.h"
#include "ExtractorType.h"
#include <chrono>

using namespace std;

AlikedRunner::AlikedRunner(const std::string& model_path, const int model_type)
{
    cfg.channels = 3;
    if (model_type == ExtractorType::aliked_n16_1k)
    {
        cfg.mEnginePath = model_path + std::string("/aliked-n16rot-top1k-tum.onnx");
        cfg.modelName = std::string("aliked-n16-top1k");
        cfg.k = 1000;
        cfg.dims = 128;
    }
    if (model_type == ExtractorType::aliked_n16_2k)
    {
        cfg.mEnginePath = model_path + std::string("/aliked-n16rot-top2k-tum.onnx");
        cfg.modelName = std::string("aliked-n16-top2k");
        cfg.k = 2000;
        cfg.dims = 128;
    }
    if (model_type == ExtractorType::aliked_n32_1k)
    {
        cfg.mEnginePath = model_path + std::string("/aliked-n32-top1k-tum.onnx");
        cfg.modelName = std::string("aliked-n32-top1k");
        cfg.k = 1000;
        cfg.dims = 128;
    }
    if (model_type == ExtractorType::aliked_n32_2k)
    {
        cfg.mEnginePath = model_path + std::string("/aliked-n32-top2k-tum.onnx");
        cfg.modelName = std::string("aliked-n32-top2k");
        cfg.k = 2000;
        cfg.dims = 128;
    }
    if (model_type == ExtractorType::superpoint)
    {
        cfg.mEnginePath = model_path + std::string("/superpoint.onnx");
        cfg.modelName = std::string("superpoint");
        cfg.k = 1000;
        cfg.dims = 256;
        cfg.channels = 1;
    }

    cfg.width = 640;
    cfg.height = 480;
    cfg.device = "cuda";
    cfg.nchw = true;

    mpOnnx = std::make_unique<AlikedOnnxManager>();

    std::cout << "Loading: " << cfg.modelName << std::endl;
    mpOnnx->init(cfg);
}

void AlikedRunner::inference(cv::Mat img)
{
    cv::Size imgSize = img.size();
    std::vector<Ort::Value> output_tensors = mpOnnx->inference(img);

    // Get pointer to output tensors
    float* pKP = output_tensors[0].GetTensorMutableData<float>();
    keypoints = cv::Mat(cfg.k, 2, CV_32F, pKP);

    float* pDescs = output_tensors[1].GetTensorMutableData<float>();
    descriptors = cv::Mat(cfg.k, cfg.dims, CV_32F, pDescs);

    float* pScores = output_tensors[2].GetTensorMutableData<float>();
    scores = {pScores, pScores + cfg.k};


    // Scale keypoints from [-1, 1] to image width and height
    cv::Mat temp_kpts = keypoints.clone();
    cv::Size kptSize = temp_kpts.size();

    cv::Vec2f* ptr;
    for (int i = 0; i < kptSize.height; i++)
    {
        ptr = temp_kpts.ptr<cv::Vec2f>(i);
        // (width - 1) * (x + 1) / 2, (height - 1) * (y + 1) / 2
        ptr[0] = cv::Vec2f((imgSize.width-1) * (ptr[0].val[0] + 1) / 2, (imgSize.height-1) * (ptr[0].val[1] + 1) / 2);
    }

    temp_kpts.convertTo(processed_keypoints, CV_16U);

}

cv::Mat AlikedRunner::getKeypoints()
{
    return keypoints.clone();
}
cv::Mat AlikedRunner::getProcessedKeypoints()
{
    return processed_keypoints.clone();
}
cv::Mat AlikedRunner::getDescriptors()
{
    return descriptors.clone();
}
std::vector<float> AlikedRunner::getScores()
{
    return std::move(scores);
}

