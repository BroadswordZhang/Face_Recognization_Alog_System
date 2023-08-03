#include <fstream>

#include "cv_dnn_centerface.h"


Centerface::Centerface(std::string model_path, int width, int height) {
//	net = cv::dnn::readNetFromONNX(model_path);

    char *trtModelStream = NULL;
    int size = 0;
    std::ifstream file(model_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    runtime = createInferRuntime(sample::gLogger.getTRTLogger());
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    CHECK(cudaMalloc(&buffers[0], MAX_SIZE_INPUT * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], MAX_SIZE_OUTPUT1 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[2], MAX_SIZE_OUTPUT2 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[3], MAX_SIZE_OUTPUT3 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[4], MAX_SIZE_OUTPUT4 * sizeof(float)));

    CHECK(cudaStreamCreate(&stream));

    input_host = new float[MAX_SIZE_INPUT * sizeof(float)];
    output1_host = new float[MAX_SIZE_OUTPUT1 * sizeof(float)];
    output2_host = new float[MAX_SIZE_OUTPUT2 * sizeof(float)];
    output3_host = new float[MAX_SIZE_OUTPUT3 * sizeof(float)];
    output4_host = new float[MAX_SIZE_OUTPUT4 * sizeof(float)];

    dynamic_scale(width, height);
}

Centerface::~Centerface() {

}

static void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output,
                        int INPUT_H, int INPUT_W, int OUTPUT_SIZE, int batchSize = 1) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice,
                          stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void Centerface::detect(cv::Mat &image, std::vector<FaceInfo> &faces, float scoreThresh, float nmsThresh) {
    image_h = image.rows;
    image_w = image.cols;
    const int batch_size = 1;

    scale_w = (float) image_w / (float) d_w;
    scale_h = (float) image_h / (float) d_h;

    cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1.0, cv::Size(d_w, d_h), cv::Scalar(0, 0, 0), true);
    input_host = (float *) inputBlob.data;

    int input_size = batch_size * 3 * inputBlob.size[2] * inputBlob.size[3] * sizeof(float);
    int output1_size = batch_size * 1 * inputBlob.size[2] * inputBlob.size[3] * sizeof(float) / 16;
    int output2_size = batch_size * 2 * inputBlob.size[2] * inputBlob.size[3] * sizeof(float) / 16;
    int output3_size = batch_size * 2 * inputBlob.size[2] * inputBlob.size[3] * sizeof(float) / 16;
    int output4_size = batch_size * 10 * inputBlob.size[2] * inputBlob.size[3] * sizeof(float) / 16;
    assert(input_size <= MAX_SIZE_INPUT);
    assert(output1_size <= MAX_SIZE_OUTPUT1);
    assert(output2_size <= MAX_SIZE_OUTPUT2);
    assert(output3_size <= MAX_SIZE_OUTPUT3);
    assert(output4_size <= MAX_SIZE_OUTPUT4);

//    auto dim1 = context->getBindingDimensions(1);
    if (!context->setBindingDimensions(0, Dims4(1, 3, inputBlob.size[2], inputBlob.size[3]))) {
        printf(" SHAPE SET ERROR ");
        exit(-1);
    } else {
        auto dim1 = context->getBindingDimensions(1);
        auto dim2 = context->getBindingDimensions(2);
        auto dim3 = context->getBindingDimensions(3);
        auto dim4 = context->getBindingDimensions(4);
    }
    CHECK(cudaMemcpyAsync(buffers[0], input_host, input_size, cudaMemcpyHostToDevice, stream));
    context->enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output1_host, buffers[1], output1_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output2_host, buffers[2], output2_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output3_host, buffers[3], output3_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output4_host, buffers[4], output4_size, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    std::vector<cv::Mat> out_blobs;

    int dims_o1[4] = {1, 1, inputBlob.size[2] / 4, inputBlob.size[3] / 4};
    out_blobs.emplace_back(cv::Mat(4, dims_o1, CV_32FC4, (void *) output1_host));
    int dims_o2[4] = {1, 2, inputBlob.size[2] / 4, inputBlob.size[3] / 4};
    out_blobs.emplace_back(cv::Mat(4, dims_o2, CV_32FC4, (void *) output2_host));
    int dims_o3[4] = {1, 2, inputBlob.size[2] / 4, inputBlob.size[3] / 4};
    out_blobs.emplace_back(cv::Mat(4, dims_o3, CV_32FC4, (void *) output3_host));
    int dims_o4[4] = {1, 10, inputBlob.size[2] / 4, inputBlob.size[3] / 4};
    out_blobs.emplace_back(cv::Mat(4, dims_o4, CV_32FC4, (void *) output4_host));


    decode(out_blobs[0], out_blobs[1], out_blobs[2], out_blobs[3], faces, scoreThresh, nmsThresh);
    squareBox(faces);
}

void Centerface::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, float nmsthreshold) {
    std::sort(input.begin(), input.end(),
              [](const FaceInfo &a, const FaceInfo &b) {
                  return a.score > b.score;
              });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;

        output.push_back(input[i]);

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;


        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;//std::max(input[i].x1, input[j].x1);
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;  //bug fixed ,sorry
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;


            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > nmsthreshold)
                merged[j] = 1;
        }

    }
}

void
Centerface::decode(cv::Mat &heatmap, cv::Mat &scale, cv::Mat &offset, cv::Mat &landmarks, std::vector<FaceInfo> &faces,
                   float scoreThresh, float nmsThresh) {
    int fea_h = heatmap.size[2];
    int fea_w = heatmap.size[3];
    int spacial_size = fea_w * fea_h;

    float *heatmap_ = (float *) (heatmap.data);

    float *scale0 = (float *) (scale.data);
    float *scale1 = scale0 + spacial_size;

    float *offset0 = (float *) (offset.data);
    float *offset1 = offset0 + spacial_size;
    float *lm = (float *) landmarks.data;

    std::vector<int> ids = getIds(heatmap_, fea_h, fea_w, scoreThresh);
    //std::cout << ids.size() << std::endl;

    std::vector<FaceInfo> faces_tmp;
    for (int i = 0; i < ids.size() / 2; i++) {
        int id_h = ids[2 * i];
        int id_w = ids[2 * i + 1];
        int index = id_h * fea_w + id_w;

        float s0 = std::exp(scale0[index]) * 4;
        float s1 = std::exp(scale1[index]) * 4;
        float o0 = offset0[index];
        float o1 = offset1[index];

        //std::cout << s0 << " " << s1 << " " << o0 << " " << o1 << std::endl;

        float x1 = std::max(0., (id_w + o1 + 0.5) * 4 - s1 / 2);
        float y1 = std::max(0., (id_h + o0 + 0.5) * 4 - s0 / 2);
        float x2 = 0, y2 = 0;
        x1 = std::min(x1, (float) d_w);
        y1 = std::min(y1, (float) d_h);
        x2 = std::min(x1 + s1, (float) d_w);
        y2 = std::min(y1 + s0, (float) d_h);

        //std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;

        FaceInfo facebox;
        facebox.x1 = x1;
        facebox.y1 = y1;
        facebox.x2 = x2;
        facebox.y2 = y2;
        facebox.score = heatmap_[index];

        //float box_w = std::min(x1 + s1, (float)d_w)-x1;
        //float box_h = std::min(y1 + s0, (float)d_h)-y1;

        float box_w = x2 - x1;
        float box_h = y2 - y1;

        //std::cout << facebox.x1 << " " << facebox.y1 << " " << facebox.x2 << " " << facebox.y2 << std::endl;

        for (int j = 0; j < 5; j++) {
            facebox.landmarks[2 * j] = x1 + lm[(2 * j + 1) * spacial_size + index] * s1;
            facebox.landmarks[2 * j + 1] = y1 + lm[(2 * j) * spacial_size + index] * s0;
            //std::cout << facebox.x1 << " " << facebox.y1 <<  std::endl;
            //std::cout << facebox.landmarks[2 * j] << " " << facebox.landmarks[2 * j+1]  << std::endl;
        }
        faces_tmp.push_back(facebox);
    }


    nms(faces_tmp, faces, nmsThresh);

    //std::cout << faces.size() << std::endl;

    for (int k = 0; k < faces.size(); k++) {
        faces[k].x1 *= d_scale_w * scale_w;
        faces[k].y1 *= d_scale_h * scale_h;
        faces[k].x2 *= d_scale_w * scale_w;
        faces[k].y2 *= d_scale_h * scale_h;

        for (int kk = 0; kk < 5; kk++) {
            faces[k].landmarks[2 * kk] *= d_scale_w * scale_w;
            faces[k].landmarks[2 * kk + 1] *= d_scale_h * scale_h;
        }
    }

}

void Centerface::dynamic_scale(float in_w, float in_h) {
    d_h = (int) (std::ceil(in_h / 32) * 32);
    d_w = (int) (std::ceil(in_w / 32) * 32);

    d_scale_h = in_h / d_h;
    d_scale_w = in_w / d_w;
}

std::vector<int> Centerface::getIds(float *heatmap, int h, int w, float thresh) {
    std::vector<int> ids;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (heatmap[i * w + j] > thresh) {
                std::array<int, 2> id = {i, j};
                ids.push_back(i);
                ids.push_back(j);
            }
        }
    }
    return ids;
}

void Centerface::squareBox(std::vector<FaceInfo> &faces) {
    float w = 0, h = 0, maxSize = 0;
    float cenx, ceny;
    for (int i = 0; i < faces.size(); i++) {
        w = faces[i].x2 - faces[i].x1;
        h = faces[i].y2 - faces[i].y1;

        maxSize = std::max(w, h);
        cenx = faces[i].x1 + w / 2;
        ceny = faces[i].y1 + h / 2;

        faces[i].x1 = std::max(cenx - maxSize / 2,
                               0.f);                 // cenx - maxSize / 2 > 0 ? cenx - maxSize / 2 : 0;
        faces[i].y1 = std::max(ceny - maxSize / 2,
                               0.f);                     //ceny - maxSize / 2 > 0 ? ceny - maxSize / 2 : 0;
        faces[i].x2 = std::min(cenx + maxSize / 2,
                               image_w - 1.f);  // cenx + maxSize / 2 > image_w - 1 ? image_w - 1 : cenx + maxSize / 2;
        faces[i].y2 = std::min(ceny + maxSize / 2,
                               image_h - 1.f); //ceny + maxSize / 2 > image_h-1 ? image_h - 1 : ceny + maxSize / 2;
    }
}


