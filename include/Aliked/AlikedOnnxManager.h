#pragma once

#include <fstream> 
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "IOnnxRunner.h"
#include "Config.h"
#include "onnxruntime_c_api.h"

using namespace std;

class AlikedOnnxManager : IOnnxRunner
{
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session* sess;
    bool session_init = false;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string inputNodeNames;

    Ort::Value input_tensor{nullptr};
    std::vector<Ort::Value> output_tensors;

    cv::Mat blob;

    OnnxConfig cfg;


public:
    AlikedOnnxManager() : IOnnxRunner() {}

    void init(const OnnxConfig& cfg_) override
    {
        cfg = cfg_;

        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, cfg.modelName.c_str());
        session_options = Ort::SessionOptions();
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (cfg.device == "cuda")
        {
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.gpu_mem_limit = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.do_copy_in_default_stream = 1;

            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }

        sess = new Ort::Session(env, cfg.mEnginePath.c_str(), session_options);

        size_t numInputNodes = sess->GetInputCount();
        if (numInputNodes != 1)
        {
            std::cerr << cfg.modelName << " has incorrect number of inputs" << std::endl;
        }

        inputNodeNames = sess->GetInputNameAllocated(0, allocator).get();

        size_t numOutputNodes= sess->GetOutputCount();
        if (numOutputNodes != 3)
        {
            std::cerr << cfg.modelName<< " has incorrect number of outputs" << std::endl;
        }

        session_init = true;
    }

    std::vector<Ort::Value> inference(cv::Mat& img)
    {
        if (img.channels() == 3 and cfg.channels == 1) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        if (img.channels() == cfg.channels)
        {
            cv::resize(img, img, cv::Size(cfg.width, cfg.height));
            img.convertTo(img, CV_32F);

            doInference(img);
        }
        else
        {
            printf("Error, image has wrong number of channels\n");
        }

        //for (int i = 0; i < output_tensors.size(); i++)
        //{
        //    auto output_info = output_tensors[i].GetTensorTypeAndShapeInfo();
        //    std::cout << "Output node index: " << i << std::endl;
        //    std::cout << "ONNX Element Type: " << output_info.GetElementType() << std::endl;
        //    std::cout << "Shape of the output: " << output_info.GetShape()[0] << "," << output_info.GetShape()[1] << std::endl;
        //}

        return std::move(output_tensors);
    }

    void doInference(cv::Mat input) override
    {
        const char* input_names[] = {inputNodeNames.c_str()};
        const char* output_names[] = {"keypoints", "descriptors", "scores"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        // Resizes image to NCHW format, swap RB set to true, crop after resize set to false
        blob = cv::dnn::blobFromImage(input, 1 / 255.0, cv::Size(cfg.width, cfg.height), (0, 0, 0), false, false);
        std::array<int64_t, 4> input_shape {1, cfg.channels, cfg.height, cfg.width};
        input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, cfg.height * cfg.width * cfg.channels * sizeof(float),
                                                        input_shape.data(), input_shape.size());
        output_tensors = sess->Run(Ort::RunOptions(nullptr), input_names, &input_tensor, 1, output_names, 3);
    }
};

