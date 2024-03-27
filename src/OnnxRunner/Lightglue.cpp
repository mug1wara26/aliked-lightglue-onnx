#pragma once

#include "Lightglue/OnnxRunner.h"
#include "ExtractorType.h"

LightglueRunner::LightglueRunner(const std::string& model_path, const int extractor_type)
{
    if (extractor_type == ExtractorType::superpoint)
    {
        cfg.mEnginePath = model_path + std::string("/superpoint_lightglue.onnx");
        cfg.modelName = std::string("superpoint_lightglue");
        cfg.num_desc = 256;
    }
    else
    {
        cfg.mEnginePath = model_path + std::string("/aliked_lightglue.onnx");
        cfg.modelName = std::string("aliked_lightglue");
        cfg.num_desc = 128;
    }

    cfg.device = "cuda";

    mpOnnx = std::make_unique<LightglueOnnxManager>();

    std::cout << "Loading: " << cfg.modelName << std::endl;
    mpOnnx->init(cfg);
}

void LightglueRunner::inference(cv::Mat kpt0, cv::Mat kpt1, cv::Mat desc0, cv::Mat desc1)
{
    std::vector<Ort::Value> output_tensors = mpOnnx->doInference(kpt0, kpt1, desc0, desc1);

    // Get number of matches from output shape
    std::vector<int64_t> matches_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t* pMatches = (int64_t*)output_tensors[0].GetTensorMutableData<int64_t>();

    // Convert tensor output from int64 to int32, Mats only go up to int32
    matches = cv::Mat(matches_shape[0], 2, CV_32S);
    cv::Vec2i* ptr;
    for (int i = 0; i < matches_shape[0]; i++)
    {
        ptr = matches.ptr<cv::Vec2i>(i);
        ptr[0] = cv::Vec2i(pMatches[i*2], pMatches[i*2+1]);
    }

    float* pScores = output_tensors[1].GetTensorMutableData<float>();
    scores = {pScores, pScores + matches_shape[0]};
}

cv::Mat LightglueRunner::getMatches()
{
    return matches.clone();
}

std::vector<float> LightglueRunner::getScores()
{
    return std::move(scores);
}
