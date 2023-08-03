import argparse
import aidlite_gpu
import math
import cv2
from skimage import transform as trans
from cvs import cvs
from Retinaface_utils import * 

def parse_opt():
    parser = argparse.ArgumentParser()
    # Default args
    parser.add_argument('--video_path', default='/codes/Face_Rec/test/videos/Kuangbiao.mp4', help='video path')
    # Retinaface args
    parser.add_argument('--retinaface_model_path', default='/codes/Face_Rec/models/mobilenet0_25.tflite', help='Retinaface model path')
    parser.add_argument('--retinaface_backbone', default='mobilenet', help='Retinaface model backbone')
    parser.add_argument('--retinaface_threshold', type=float, default=0.5, help='Retinaface detect threshold')
    parser.add_argument('--retinaface_nms_iou', type=float,  default=0.45, help='Retinaface nms iou threshold')
    parser.add_argument('--retinaface_input_shape', type=int,default=[640, 640, 3], help='Retinaface model backbone')
    parser.add_argument('--retinaface_output_shape', type=int,default=[10, 2, 4], help='Retinaface model output shape')

    opt = parser.parse_args()
    print(opt)
    return opt

def model_init(opt):
    Aidlite_Engine = aidlite_gpu.aidlite()
    # Aidlux Retinaface model init
    Aidlite_Engine.set_g_index(0)
    model_path_retina = opt.retinaface_model_path
    in_shape_retina = [1 * 640 * 640 * 3 * 4]
    out_shape_retina = [1 * 16800 * 10 * 4, 1 * 16800 * 2 * 4, 1 * 16800 * 4 * 4]
    # 载入retinaface mobilenet检测模型
    res1 = Aidlite_Engine.ANNModel(model_path_retina, in_shape_retina, out_shape_retina, 4, 0)
    print("retinaface model init: ", res1)
    return Aidlite_Engine

def euler_angle_judge(raw_image, img_size, face_dets):
    # Camera internals
    center = (img_size[1] / 2, img_size[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    model_points = np.array([(0.0, 0.0, 0.0), (-125.0, 170.0, -135.0), (125.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
     # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))
    face_dets_to_align = []
    for i in range(len(face_dets)):
        landmarks = face_dets[i][5:15]
        image_points = np.array([(landmarks[4], landmarks[5]), (landmarks[0], landmarks[1]),  (landmarks[2], landmarks[3]), (landmarks[6], landmarks[7]),  (landmarks[8], landmarks[9])  ], dtype="double")
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        # 俯仰角在正对的时候是-180度，需要转换成锐角
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(roll)
        yaw = math.degrees(yaw)

        #俯仰角和偏航角大于30度的人脸过滤掉，任何旋转角度都可以进入人脸矫正
        if pitch.__abs__() < 30 and yaw.__abs__() < 30:
                face_dets_to_align.append(face_dets[i])

    return raw_image, np.array(face_dets_to_align)

def align_face(img, face_dets_to_align):
    aligned_faces = []  # 对齐的人脸数据，112x112
    boxes = []
    landmarks = []
    if len(face_dets_to_align):
        boxes = face_dets_to_align[:, :4]
        landmarks = face_dets_to_align[:, 5:15]
        landmarks = landmarks.reshape(len(landmarks), 5, 2)
        for i in range(len(boxes)):
            landmark = landmarks[i]
            aligned_face_img = align_process(img, landmark)
            # debug
            aligned_test_image = cv2.resize(aligned_face_img, (336, 336))
            img[0:336, 0:336] = aligned_test_image
    return img

def align_process(img, landmark=None):
        M = None
        image_size = [112, 112]
        if landmark is not None:
            assert len(image_size) == 2
            src = np.array([[30.2946, 51.6963],[65.5318, 51.5014],[48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041]], dtype=np.float32)
            src[:, 0] += 8.0
            dst = landmark.astype(np.float32)
            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2, :]
            assert len(image_size) == 2
            warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
            return warped

def main(opt):
    Face_Detector = Retinaface(opt)
    Aidlite_Engine = model_init(opt)
    cap = cvs.VideoCapture(opt.video_path)
    fps = 0.0
    frame_id = 0
    while (True):
        raw_frame = cap.read()
        frame_id += 1
        if frame_id % 6 != 0:
            continue
        retinaface_image, face_dets = Face_Detector.detect_image(raw_frame, Aidlite_Engine)
        euler_angle_image, face_dets_to_align = euler_angle_judge(raw_frame, raw_frame.shape, face_dets)
        Aligned_image = align_face(retinaface_image, face_dets_to_align)

        cvs.imshow(Aligned_image)

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)