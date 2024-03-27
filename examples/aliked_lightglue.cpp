#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Aliked/OnnxRunner.h"
#include "Lightglue/OnnxRunner.h"

using namespace cv;
using namespace std;

void plotMatches(cv::Mat img0, cv::Mat img1, cv::Mat kpts0, cv::Mat kpts1, 
        std::vector<float> scores0, std::vector<float> scores1, cv::Mat matches, 
        std::vector<float> m_scores, float thresh)
{
    float kpts_thresh = 0;

    for (int i = 0; i < kpts0.rows; i++)
    {
        if (scores0[i] > kpts_thresh)
        {
            unsigned short x = kpts0.at<unsigned short>(i, 0);
            unsigned short y = kpts0.at<unsigned short>(i, 1);

            cv::circle(img0, cv::Point(x, y), 1, cv::Scalar(0,255,0), -1, 8, 0);
        }
    }
    for (int i = 0; i < kpts1.rows; i++)
    {
        if (scores1[i] > kpts_thresh)
        {
            unsigned short x = kpts1.at<unsigned short>(i, 0);
            unsigned short y = kpts1.at<unsigned short>(i, 1);

            cv::circle(img1, cv::Point(x, y), 1, cv::Scalar(0,255,0), -1, 8, 0);
        }
    }

    cv::Mat vis;
    cv::hconcat(img0, img1, vis);

    int count = 0;
    for (int i = 0; i < matches.rows; i++)
    {
        if (m_scores[i] > thresh)
        {
            int kpt0_i = matches.at<int>(i,0);
            int kpt1_i = matches.at<int>(i,1);

            if (scores0[kpt0_i] > kpts_thresh && scores1[kpt1_i] > kpts_thresh)
            {
                count++;
                cv::Point kpt0(kpts0.at<unsigned short>(kpt0_i,0), kpts0.at<unsigned short>(kpt0_i,1));
                cv::Point kpt1(kpts1.at<unsigned short>(kpt1_i,0) + img0.cols, kpts1.at<unsigned short>(kpt1_i,1));

                cv::line(vis, kpt0, kpt1, cv::Scalar(0,255,0), 1, cv::LINE_8);
            }
        }
    }
 
    std::cout << "Num matches: " << count << std::endl;

    cv::imshow("Display window", vis);
    cv::waitKey();
}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cout << "Usage: ./aliked_lightglue model_dir image_dir0 lmage_dir1 aliked_model_type match_thresh" << std::endl;
        return -1;
    }

    const string model_dir = string(argv[1]);
    const string image_dir0 = string(argv[2]);
    const string image_dir1 = string(argv[3]);
    const int model_type = strtol(argv[4], NULL, 10);
    const float thresh = stof(argv[5]);

    vector<cv::String> fn1;
    glob(image_dir0 + string("/*.png"), fn1, false);
    vector<cv::String> fn2;
    glob(image_dir1 + string("/*.png"), fn2, false);

    //cv::Mat image0 = imread(image0_path);
    //if(image0.empty())
    //{
    //    std::cout << "Could not read the image: " << image0_path << std::endl;
    //    return 1;
    //}

    //cv::Mat image1 = imread(image1_path);
    //if(image1.empty())
    //{
    //    std::cout << "Could not read the image: " << image1_path << std::endl;
    //    return 1;
    //}

    auto aliked = std::shared_ptr<AlikedRunner>(
            new AlikedRunner(model_dir, model_type)
            );
    auto lightglue = std::shared_ptr<LightglueRunner>(
            new LightglueRunner(model_dir, model_type)
            );

    for (int i = 0; i < fn1.size(); i++)
    {
        cv::Mat image0 = cv::imread(fn1[i]);
        cv::Mat image1 = cv::imread(fn2[i]);


        aliked->inference(image0);
        cv::Mat kpts0 = aliked->getKeypoints();
        cv::Mat processed_kpts0 = aliked->getProcessedKeypoints();
        cv::Mat desc0 = aliked->getDescriptors();
        std::vector<float> scores0 = aliked->getScores();

        aliked->inference(image1);
        cv::Mat kpts1 = aliked->getKeypoints();
        cv::Mat processed_kpts1 = aliked->getProcessedKeypoints();
        cv::Mat desc1 = aliked->getDescriptors();
        std::vector<float>scores1 = aliked->getScores();

        lightglue->inference(kpts0, kpts1, desc0, desc1);

        cv::Mat matches = lightglue->getMatches();
        std::vector<float> m_scores = lightglue->getScores();

        plotMatches(image0.clone(), image1.clone(), processed_kpts0, processed_kpts1, scores0, scores1, matches, m_scores, thresh);
    }

    return 0;
}

