import argparse
# import aidlite_gpu
import cv2
import onnxruntime
# from cvs import cvs
from Retinaface_utils import *
from FaceRecognizer_utils import *


def parse_opt():
    parser = argparse.ArgumentParser()
    # Default args
    parser.add_argument('--video_path', default='test/videos/Kuangbiao2.mp4', help='video path')
    parser.add_argument('--frame_skip', type=int, default=2, help='Frame skip num')
    # Retinaface args
    parser.add_argument('--retinaface_model_path', default='/codes/Face_Rec/models/mobilenet0_25.tflite',
                        help='Retinaface model path')
    parser.add_argument('--retinaface_backbone', default='mobilenet', help='Retinaface model backbone')
    parser.add_argument('--retinaface_threshold', type=float, default=0.7, help='Retinaface detect threshold')
    parser.add_argument('--retinaface_nms_iou', type=float, default=0.45, help='Retinaface nms iou threshold')
    parser.add_argument('--retinaface_input_shape', type=int, default=[640, 640, 3], help='Retinaface input shape')
    parser.add_argument('--retinaface_output_shape', type=int, default=[10, 2, 4], help='Retinaface model output shape')
    # FaceOperator args
    parser.add_argument('--euler_angle_threshold', default=40, type=float, help='Filter euler angle threshold')
    # Arcface args
    parser.add_argument('--arcface_model_path', default='/codes/Face_Rec/models/arcface.tflite',
                        help='Arcface model path')
    parser.add_argument('--arcface_input_shape', type=int, default=[112, 112, 3], help='Arcface model input size')
    parser.add_argument('--arcface_output_shape', type=int, default=[128], help='Arcface extract feature shape')
    # Faiss args
    parser.add_argument('--gallery_path', default='Face_Gallery/gallery_kuangbiao_pc.csv',
                        help='Face features gallery csv file')
    parser.add_argument('--face_rec_threshold', type=float, default=0.6, help='Face recognition threshold')

    opt = parser.parse_args()
    print(opt)
    return opt


def AidliteEngine_init(retinaface_opt, arcface_opt):
    Aidlite_Engine = aidlite_gpu.aidlite()
    # Aidlux Retinaface model init
    Aidlite_Engine.set_g_index(0)
    model_path_retina = retinaface_opt.retinaface_model_path
    in_shape_retina = retinaface_opt.retinaface_input_shape
    out_shape_retina = retinaface_opt.retinaface_output_shape
    # Retinaface mobilenet
    res1 = Aidlite_Engine.ANNModel(model_path_retina, in_shape_retina, out_shape_retina, 4, 0)
    print("retinaface model init result: ", res1)

    # Aidlux Arcface model init
    Aidlite_Engine.set_g_index(1)
    model_path_arcface = arcface_opt.arcface_model_path
    in_shape_arcface = arcface_opt.arcface_input_shape
    out_shape_arcface = arcface_opt.arcface_output_shape
    res2 = Aidlite_Engine.ANNModel(model_path_arcface, in_shape_arcface, out_shape_arcface, 4, 0)
    print("Arcface model init result: ", res2)

    return Aidlite_Engine

def model_init_pc():
    weights = 'models/mobilenet0_25.onnx'
    sess = onnxruntime.InferenceSession(weights)
    input_name = sess.get_inputs()[0].name
    output_names = []
    for i in range(len(sess.get_outputs())):
        # print('output shape retinaface:', sess.get_outputs()[i].name)
        output_names.append(sess.get_outputs()[i].name)

    output_name = sess.get_outputs()[0].name
    # print('input name:%s, output name:%s' % (input_name, output_name))
    input_shape = sess.get_inputs()[0].shape
    print('input_shape:', input_shape)
    retina_model = sess

    arcface_path = 'models/arcface/model-y1/model,0'
    arcface_layer ='fc1'
    ctx = mxnet.cpu()
    _vec = arcface_path.split(',')
    prefix = _vec[0]
    epoch = int(_vec[1])
    # 输入的symbol:网络名称，一个NDArray字典，以及网络权重字典 NDArray 模型参数以及一些附加状态的字典
    sym, arg_params, aux_params = mxnet.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[arcface_layer + '_output']
    arcface_model = mxnet.mod.Module(symbol=sym, context=ctx, label_names=None)
    # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    # 绑定输入数据的形状，分配内存
    arcface_model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    # 将训练好的参数进行赋值
    arcface_model.set_params(arg_params, aux_params)

    return retina_model, arcface_model

def main(opt):
    # init
    FaceDetector = Retinaface(opt)
    FaceOperator = Face_Operator(opt)
    FaceExtractor = Arcface(opt)
    FaissEngine = Faiss_Engine(opt)

    # AidliteEngine = AidliteEngine_init(FaceDetector, FaceExtractor)
    RetinaModel, ArcfaceModel = model_init_pc()

    cap = cv2.VideoCapture(opt.video_path)
    fps = 0.0
    frame_id = 0
    while (True):
        ok, raw_frame = cap.read()
        raw_image = raw_frame.copy()
        frame_id += 1
        if frame_id % opt.frame_skip != 0:
            continue
        # 人脸关键点检测
        retinaface_image, face_dets = FaceDetector.detect_image_onnx(raw_frame, RetinaModel)
        # Arcface preprocess
        if len(face_dets):
            # 头部姿态估计过滤不理想的人脸
            euler_image, face_dets_to_align = FaceOperator.euler_judge(raw_frame, raw_frame.shape, face_dets)
            # 人脸矫正对齐
            aligned_faces, boxes, landmarks = FaceOperator.align_face(raw_frame, face_dets_to_align)
            frame_features = []
            for aligned_face in aligned_faces:
                # 人脸特征提取
                feature = FaceExtractor.extract_image_mxnet(ArcfaceModel, aligned_face)
                frame_features.append(feature)
                # faiss库搜索并标记人名
            raw_image = FaissEngine.face_search(frame_features, raw_frame, boxes, landmarks)
        cv2.imshow('result', raw_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
