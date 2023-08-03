#pragma once

#include<string>
#include<vector>
#include<algorithm>
#include <numeric>
#include<math.h>
#include<opencv2/opencv.hpp>

#include "logger.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "common.h"

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float landmarks[10];
};

using namespace nvinfer1;

class Centerface {
public:
    Centerface(std::string model_path, int width, int height);

    ~Centerface();

    void detect(cv::Mat &image, std::vector<FaceInfo> &faces, float scoreThresh = 0.5, float nmsThresh = 0.3);

private:
    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, float nmsthreshold = 0.3);

    void decode(cv::Mat &heatmap, cv::Mat &scale, cv::Mat &offset, cv::Mat &landmarks, std::vector<FaceInfo> &faces,
                float scoreThresh, float nmsThresh);

    void dynamic_scale(float in_w, float in_h);

    std::vector<int> getIds(float *heatmap, int h, int w, float thresh);

    void squareBox(std::vector<FaceInfo> &faces);

private:
    int d_h;
    int d_w;
    float d_scale_h;
    float d_scale_w;

    float scale_w;
    float scale_h;

    int image_h;
    int image_w;

    IRuntime *runtime = NULL;
    ICudaEngine *engine = NULL;
    IExecutionContext *context = NULL;

    static const int MAX_BATCH_SIZE = 1;
    static const int MAX_INPUT_H = 544;
    static const int MAX_INPUT_W = 960;
    static const int MAX_SIZE_INPUT = MAX_BATCH_SIZE * 3 * MAX_INPUT_H * MAX_INPUT_W * sizeof(float);

    static const int MAX_SIZE_OUTPUT1 = MAX_BATCH_SIZE * 1 * MAX_INPUT_H * MAX_INPUT_W / 16 * sizeof(float);
    static const int MAX_SIZE_OUTPUT2 = MAX_BATCH_SIZE * 2 * MAX_INPUT_H * MAX_INPUT_W / 16 * sizeof(float);
    static const int MAX_SIZE_OUTPUT3 = MAX_BATCH_SIZE * 2 * MAX_INPUT_H * MAX_INPUT_W / 16 * sizeof(float);
    static const int MAX_SIZE_OUTPUT4 = MAX_BATCH_SIZE * 10 * MAX_INPUT_H * MAX_INPUT_W / 16 * sizeof(float);

    float *input_host = NULL;
    float *output1_host = NULL;
    float *output2_host = NULL;
    float *output3_host = NULL;
    float *output4_host = NULL;

    void **buffers = (void **) malloc(sizeof(float *));

    cudaStream_t stream;

    cv::dnn::Net net;
};


