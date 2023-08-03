#include <iostream>
#include <opencv2/opencv.hpp>
#include "cpp/cv_dnn_centerface.h"

#include "sys/time.h"

int main(int argc, char **argv) {
    std::string model_path;
    std::string image_file;
    if (argc == 3) {

        model_path = argv[1];
        image_file = argv[2];
    } else {
        std::cout << " .exe mode_path image_file, now  using default" << std::endl;
//        model_path = "../../models/onnx/centerface.onnx";
//        model_path = "../../models/tensorrt/centerface.trt";
        model_path = "../../models/tensorrt/centerface.int8.trt";
        image_file = "../../prj-python/000388.jpg";
    }

    const int profile_w = 960;
    const int profile_h = 544;
    Centerface centerface(model_path, profile_w, profile_h);

    cv::Mat image = cv::imread(image_file);
    std::vector<FaceInfo> face_info;

    timeval t0, t1;
    gettimeofday(&t0, NULL);
    for (int j = 0; j < 1000; j++) {
        centerface.detect(image, face_info);
    }
    gettimeofday(&t1, NULL);
    printf(" time cost: %f \n", t1.tv_sec - t0.tv_sec + (t1.tv_usec - t0.tv_usec) / 1000000.0);

    for (int i = 0; i < face_info.size(); i++) {
        cv::rectangle(image, cv::Point(face_info[i].x1, face_info[i].y1), cv::Point(face_info[i].x2, face_info[i].y2),
                      cv::Scalar(0, 255, 0), 2);
        for (int j = 0; j < 5; j++) {
            cv::circle(image, cv::Point(face_info[i].landmarks[2 * j], face_info[i].landmarks[2 * j + 1]), 2,
                       cv::Scalar(255, 255, 0), 2);
        }
    }

    cv::imwrite("../out_cpp_tensorrt.jpg", image);
//    cv::imshow("test", image);
//    cv::waitKey();

    return 0;
}
