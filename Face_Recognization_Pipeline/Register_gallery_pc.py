import argparse
# import aidlite_gpu
import os
# from cvs import cvs
import onnxruntime

from Retinaface_utils import *
from FaceRecognizer_utils import *


def parse_opt():
    parser = argparse.ArgumentParser()
    # Retinaface args
    parser.add_argument('--retinaface_model_path', default='/codes/Face_Rec/models/mobilenet0_25.tflite',
                        help='Retinaface model path')
    parser.add_argument('--retinaface_backbone', default='mobilenet', help='Retinaface model backbone')
    parser.add_argument('--retinaface_threshold', type=float, default=0.7, help='Retinaface detect threshold')
    parser.add_argument('--retinaface_nms_iou', type=float, default=0.45, help='Retinaface nms iou threshold')
    parser.add_argument('--retinaface_input_shape', type=int, default=[640, 640, 3], help='Retinaface input shape')
    parser.add_argument('--retinaface_output_shape', type=int, default=[10, 2, 4], help='Retinaface model output shape')
    # FaceOperator args
    parser.add_argument('--euler_angle_threshold', default=30, type=float, help='Filter euler angle threshold')
    # Arcface args
    parser.add_argument('--arcface_model_path', default='/codes/Face_Rec/models/arcface.tflite',
                        help='Arcface model path')
    parser.add_argument('--arcface_input_shape', type=int, default=[112, 112, 3], help='Arcface model input size')
    parser.add_argument('--arcface_output_shape', type=int, default=[128], help='Arcface extract feature shape')
    # Register
    parser.add_argument('--register_csv_path', default='Face_Gallery/gallery_kuangbiao_pc.csv',
                        help='Face features gallery csv file')
    parser.add_argument('--register_images_path', default='Face_Gallery/face_images_kuangbiao',
                        help='Face images to register')

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
    retinaface_weights = 'models/mobilenet0_25.onnx'
    sess = onnxruntime.InferenceSession(retinaface_weights)
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

    arcface_weights = 'models/arcface/model-y1/model,0'
    arcface_layer ='fc1'
    ctx = mxnet.cpu()
    _vec = arcface_weights.split(',')
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

def get_face_images(opt):
    image_paths = []
    path_exp = os.path.expanduser(opt.register_images_path)
    labels = os.listdir(path_exp)
    num_classes = len(labels)
    for i in range(num_classes):
        class_name = labels[i]
        facedir = os.path.join(path_exp, class_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            image_path = [os.path.join(facedir, img) for img in images]
        image_paths = image_paths + image_path
    return image_paths, labels


def main(opt):
    # init
    FaceDetector = Retinaface(opt)
    FaceOperator = Face_Operator(opt)
    FaceExtractor = Arcface(opt)

    # AidliteEngine = AidliteEngine_init(FaceDetector, FaceExtractor)
    RetinaModel, ArcfaceModel = model_init_pc()

    image_paths, lables = get_face_images(opt)

    with open(opt.register_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for tmp_path in image_paths:
            person_name = os.path.basename(tmp_path).split('_')[0]
            face_image = cv2.imread(tmp_path)
            retinaface_image, face_dets = FaceDetector.detect_image_onnx(face_image, RetinaModel)
            # Arcface preprocess
            if len(face_dets):
                euler_image, face_dets_to_align = FaceOperator.euler_judge(face_image, face_image.shape, face_dets)
                aligned_faces, boxes, landmarks = FaceOperator.align_face(face_image, face_dets_to_align)
                if len(boxes) == 1:
                    for aligned_face in aligned_faces:
                        feature = FaceExtractor.extract_image_mxnet(ArcfaceModel, aligned_face)
                        feature = np.insert(feature.astype(np.str_), 0, person_name)  # person name插入第一列
                        writer.writerow(feature.tolist())  # person和特征值写入csv
                        print('{} register success!'.format(person_name))
                else:
                    print('{} face number or euler angle invalid, face number is {}, face euler angle must < {}'.format(
                        tmp_path, len(boxes), opt.euler_angle_threshold))
    print('end')
    os._exit(0)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
