#pragma once

#include <fstream> 
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ILightglueManager.h"
#include "LightglueConf.h"
#include "onnxruntime_c_api.h"

class LightglueOnnxManager : ILightglueManager
{
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session* sess;
    bool session_init = false;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string inputNodeNames;

    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;

    LightglueConf cfg;

public:
    LightglueOnnxManager() : ILightglueManager() {}

    void init(const LightglueConf& cfg_) override
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
        if (numInputNodes != 4)
        {
            std::cerr << cfg.modelName << " has incorrect number of inputs" << std::endl;
        }

        inputNodeNames = sess->GetInputNameAllocated(0, allocator).get();

        size_t numOutputNodes= sess->GetOutputCount();
        if (numOutputNodes != 2)
        {
            std::cerr << cfg.modelName<< " has incorrect number of outputs" << std::endl;
        }

        session_init = true;
    }

    std::vector<Ort::Value> doInference(cv::Mat kpts0, cv::Mat kpts1, cv::Mat desc0, cv::Mat desc1) override
    {
        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0", "mscores0"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);


        std::array<int64_t, 3> kpts0_shape {1, kpts0.rows, 2};
        Ort::Value kpts0_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)kpts0.data, kpts0.total(),
                kpts0_shape.data(), kpts0_shape.size());

        std::array<int64_t, 3> kpts1_shape {1, kpts1.rows, 2};
        Ort::Value kpts1_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)kpts1.data, kpts1.total(),
                kpts1_shape.data(), kpts1_shape.size());

        cv::Mat padded_desc0;
        cv::copyMakeBorder(desc0, padded_desc0, 0, 0, 0, cfg.num_desc - desc0.cols, cv::BORDER_CONSTANT, cv::Scalar(1));

        std::array<int64_t, 3> desc0_shape {1, desc0.rows, cfg.num_desc};
        Ort::Value desc0_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)padded_desc0.data, padded_desc0.total(),
                desc0_shape.data(), desc0_shape.size());

        cv::Mat padded_desc1;
        cv::copyMakeBorder(desc1, padded_desc1, 0, 0, 0, cfg.num_desc - desc1.cols, cv::BORDER_CONSTANT, cv::Scalar(1));

        std::array<int64_t, 3> desc1_shape {1, desc1.rows, cfg.num_desc};
        Ort::Value desc1_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)padded_desc1.data, padded_desc1.total(),
                desc1_shape.data(), desc1_shape.size());


        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(kpts0_tensor));
        ort_inputs.push_back(std::move(kpts1_tensor));
        ort_inputs.push_back(std::move(desc0_tensor));
        ort_inputs.push_back(std::move(desc1_tensor));

        output_tensors = sess->Run(Ort::RunOptions(nullptr), input_names, ort_inputs.data(), 4, output_names, 2);

        //for (int i = 0; i < output_tensors.size(); i++)
        //{
        //    auto output_info = output_tensors[i].GetTensorTypeAndShapeInfo();
        //    std::cout << "Output node index: " << i << std::endl;
        //    std::cout << "ONNX Element Type: " << output_info.GetElementType() << std::endl;
        //    std::cout << "Shape of the output: " << output_info.GetShape()[0] << "," << output_info.GetShape()[1] << std::endl;
        //}


        return std::move(output_tensors);
    }
};

