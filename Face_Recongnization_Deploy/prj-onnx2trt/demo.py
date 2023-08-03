import os, sys

import onnx
import pycuda.driver as cuda
import tensorrt as trt
from calibrator import CenterFaceEntropyCalibrator

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine_onnx(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) \
            as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        # config.max_workspace_size = 1 << 30  # 1GB
        # builder.max_batch_size = 1
        # builder.fp16_mode = True
        profile = builder.create_optimization_profile()
        profile.set_shape('input.1', (1, 3, 32, 32), (1, 3, 480, 480), (1, 3, 544, 960))
        config.add_optimization_profile(profile)

        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if parser.parse(model.read()) is False:
                print('parsing of ONNX file Failed ')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print('Completed parsing of ONNX file')

        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        # network.get_input(0).shape = [1, 3, max_H, max_W] #use while in static input

        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
        if os.path.exists(os.path.dirname(engine_file_path)) is False:
            os.makedirs(os.path.dirname(engine_file_path))
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine


def build_engine_onnx_int8(onnx_file_path, engine_file_path, dynamic_shape=False):
    calib = CenterFaceEntropyCalibrator("../calibration_ims", cache_file="calibration_centerface.cache")

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) \
            as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        # builder.fp16_mode = True
        # use while generating quantitative model
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib

        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if parser.parse(model.read()) is False:
                print('parsing of ONNX file Failed ')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print('Completed parsing of ONNX file')

        print('Building an engine of INT8 from file {}; this may take a while...'.format(onnx_file_path))
        if dynamic_shape:
            # optimization dimension should be same as the calibration resolution
            profile = builder.create_optimization_profile()
            profile.set_shape('input.1', (1, 3, 32, 32), (1, 3, 544, 960), (1, 3, 544, 960))
            config.add_optimization_profile(profile)
            config.set_calibration_profile(profile)
        else:
            network.get_input(0).shape = [1, 3, 544, 960]  # use while in static input

        engine = builder.build_engine(network, config)
        print("Completed creating Engine of INT8")
        if os.path.exists(os.path.dirname(engine_file_path)) is False:
            os.makedirs(os.path.dirname(engine_file_path))
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine


def build_engine_caffe_int8(deploy_file, model_file, engine_file_path):
    calib = CenterFaceEntropyCalibrator("../calibration_ims", cache_file="calibration_centerface.cache")

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.CaffeParser() as parser:

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        # builder.fp16_mode = True
        # use while generating quantitative model
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib

        # optimization dimension should be same as the calibration resolution
        profile = builder.create_optimization_profile()
        profile.set_shape('input.1', (1, 3, 32, 32), (1, 3, 544, 960), (1, 3, 544, 960))
        config.add_optimization_profile(profile)
        config.set_calibration_profile(profile)

        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=trt.float32)

        for name in ["537", "538", "539", '540']:
            network.mark_output(model_tensors.find(name))

        engine = builder.build_engine(network, config)
        print("Completed creating Engine of INT8")
        if os.path.exists(os.path.dirname(engine_file_path)) is False:
            os.makedirs(os.path.dirname(engine_file_path))
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine


def static2dynamic():
    model = onnx.load("../models/onnx/centerface.onnx")
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    onnx.save(model, "../models/onnx/centerface.d.onnx")


if __name__ == '__main__':
    static2dynamic()
    build_engine_onnx("../models/onnx/centerface.d.onnx", "../models/tensorrt/centerface.trt")
    print("==================================================================================================")
    build_engine_onnx_int8("../models/onnx/centerface.d.onnx", "../models/tensorrt/centerface.int8.trt")
    # build_engine_caffe_int8("../models/caffe/centerface.prototxt", "../models/caffe/centerface.caffemodel",
    #                         "../models/tensorrt/centerface.int8.trt")
