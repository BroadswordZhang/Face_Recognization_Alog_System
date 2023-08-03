import argparse
import aidlite_gpu
import math
import cv2
import os
from cvs import cvs
from Retinaface_utils import * 

def parse_opt():
    parser = argparse.ArgumentParser()
    # Default args
    parser.add_argument('--video_path', default='/codes/Face_Rec/test/videos/Kuangbiao.mp4', help='video path')
    # parser.add_argument('--video_path', default=1, help='video path')
    # Retinaface args
    parser.add_argument('--retinaface_model_path', default='/codes/Face_Rec/models/mobilenet0_25.tflite', help='Retinaface model path')
    parser.add_argument('--retinaface_backbone', default='mobilenet', help='Retinaface model backbone')
    parser.add_argument('--retinaface_threshold', type=float, default=0.5, help='Retinaface detect threshold')
    parser.add_argument('--retinaface_nms_iou', type=float,  default=0.45, help='Retinaface nms iou threshold')
    parser.add_argument('--retinaface_input_shape', type=int,default=[640, 640, 3], help='Retinaface input shape')
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
    res1 = Aidlite_Engine.ANNModel(model_path_retina, in_shape_retina, out_shape_retina, 3, 0)
    print("retinaface model init: ", res1)
    return Aidlite_Engine

def euler_angle_judge(raw_image, img_size, face_dets):
    # Camera internals
        center = (img_size[1] / 2, img_size[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        model_points = np.array([(0.0, 0.0, 0.0), (-125.0, 170.0, -135.0), (125.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        face_dets_to_align = []
        for i in range(len(face_dets)):
            landmarks = face_dets[i][5:15]
            image_points = np.array([
                (landmarks[4], landmarks[5]),  # Nose tip
                (landmarks[0], landmarks[1]),  # Left eye center
                (landmarks[2], landmarks[3]),  # Right eye center
                (landmarks[6], landmarks[7]),  # Left Mouth corner
                (landmarks[8], landmarks[9])  # Right mouth corner
            ], dtype="double")
            # 求解旋转向量，通过罗德里格公式转换为旋转矩阵
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

            # 分解旋转矩阵，获得欧拉角
            proj_matrix = np.hstack((rvec_matrix, translation_vector))
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

            pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
             # 俯仰角在正对的时候是-180度，需要转换成锐角
            pitch = math.degrees(math.asin(math.sin(pitch)))
            roll = math.degrees(roll)
            yaw = math.degrees(yaw)
            # debug
            print("euler angle pitch:{:.4f}, yaw:{:.4f}, roll:{:.4f}".format(pitch, yaw, roll))
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            for p in image_points:
                cv2.circle(raw_image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            text = 'pitch:{:.0f} roll:{:.0f} yaw:{:.0f}'.format(pitch, yaw, roll)
            cv2.putText(raw_image, text, (0, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(raw_image, p1, p2, (255, 0, 0), 2)
            return raw_image


def main(opt):
    FaceDetector = Retinaface(opt)
    AidliteEngine = model_init(opt)
    cap = cvs.VideoCapture(opt.video_path)
    fps = 0.0
    frame_id = 0
    while (True):
        raw_frame = cap.read()
        frame_id += 1
        if frame_id % 6 != 0:
            continue
        if raw_frame is None:
            print("cap read over!")
            continue
        else:
            retinaface_image, face_dets = FaceDetector.detect_image(raw_frame, AidliteEngine)
            euler_angle_image = euler_angle_judge(raw_frame,retinaface_image.shape, face_dets)
        cvs.imshow(euler_angle_image)

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
