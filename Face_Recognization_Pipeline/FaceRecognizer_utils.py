import mxnet
import numpy as np
from sklearn import preprocessing
import cv2
import math
from skimage import transform
import faiss
import csv


class Arcface():
    def __init__(self, opt):
        # aidlux
        self.arcface_model_path = opt.arcface_model_path
        self.arcface_input_shape = [1 * opt.arcface_input_shape[0] * opt.arcface_input_shape[1] * opt.arcface_input_shape[2] * 4]
        self.arcface_output_shape = [1 * opt.arcface_output_shape[0] * 4]
    
    def extract_image(self, AidliteEngine, image_input):
        img_aid_input = np.float32(np.expand_dims(image_input, axis=0))
        # invoke
        AidliteEngine.set_g_index(1)
        AidliteEngine.setInput_Float32(img_aid_input, 112, 112)
        AidliteEngine.invoke()
        feature = AidliteEngine.getOutput_Float32(0)[None, :]
        feature = preprocessing.normalize(feature).flatten()
        return feature

    def extract_image_onnx(self, OnnxEngine, image_input):
        img_aid_input = np.float32(np.expand_dims(image_input, axis=0))
        # invoke
        img = img_aid_input.reshape(1, 3, 112, 112)
        # img = img.astype(np.float32)
        feature = OnnxEngine.run(None, {'data': img})
        feature = preprocessing.normalize(feature).flatten()
        return feature

    def extract_image_mxnet(self, MxnetEngine, aligned_face):
        aligned_face = np.transpose(aligned_face, (2, 0, 1))
        # aligned_face = aligned_face.reshape(3, 112, 112)
        input_blob = np.expand_dims(aligned_face, axis=0)
        data = mxnet.nd.array(input_blob)
        db = mxnet.io.DataBatch(data=(data,))
        MxnetEngine.forward(db, is_train=False)
        embedding = MxnetEngine.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding

class Face_Operator():
    def __init__(self, opt):
        # 3D model points.
        self.model_points = np.array([(0.0, 0.0, 0.0), (-125.0, 170.0, -135.0), (125.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
        self.rec_angle = opt.euler_angle_threshold 
        self.camera_matrix = None
    
    def camera_internals(self, img_size):
        # Camera internals
        center = (img_size[1] / 2, img_size[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        self.camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    def euler_judge(self, raw_image, img_size, face_dets):
        self.camera_internals(img_size)
        # Assuming no lens distortion假设没有径向畸变
        dist_coeffs = np.zeros((4, 1))  
        face_dets_to_align = []
        for i in range(len(face_dets)):
            landmarks = face_dets[i][5:15]
            image_points = np.array([(landmarks[4], landmarks[5]), (landmarks[0], landmarks[1]),  (landmarks[2], landmarks[3]),  (landmarks[6], landmarks[7]), (landmarks[8], landmarks[9])], dtype="double")
            # 求解旋转向量，通过罗德里格公式转换为旋转矩阵
            (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
            # 分解旋转矩阵，获得欧拉角
            proj_matrix = np.hstack((rvec_matrix, translation_vector))
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

            pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
             # 俯仰角在正对的时候是-180度，需要转换成锐角
            pitch = math.degrees(math.asin(math.sin(pitch)))
            roll = -math.degrees(roll)
            yaw = math.degrees(yaw)
            if pitch.__abs__() < self.rec_angle and yaw.__abs__() < self.rec_angle:
                face_dets_to_align.append(face_dets[i])

            # debug
            # print("euler angle pitch:{:.4f}, yaw:{:.4f}, roll:{:.4f}".format(pitch, yaw, roll))
        return raw_image, np.array(face_dets_to_align)

    def align_face(self, img, face_dets_to_align):
        # 对齐的人脸数据，112x112
        aligned_faces = []  
        boxes = []
        landmarks = []
        if len(face_dets_to_align):
            boxes = face_dets_to_align[:, :4]
            landmarks = face_dets_to_align[:, 5:15]
            landmarks = landmarks.reshape(len(landmarks), 5, 2)
            for i in range(len(boxes)):
                landmark = landmarks[i]
                aligned_face_img = self.align_process(img, [112, 112], landmark)
                # debug
                # cv2.imwrite("/codes/Face_Rec/arcface_test_xzw.jpg", aligned_face_img)
                aligned_faces.append(aligned_face_img)
        return aligned_faces, np.array(boxes), np.array(landmarks)

    def align_process(self, img, image_size, landmark=None):
            M = None
            src = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]], dtype=np.float32)
            src[:, 0] += 8.0
            dst = landmark.astype(np.float32)
            tform = transform.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2, :]
            warped = cv2.warpAffine(img, M, (image_size[0], image_size[1]), borderValue=0.0)
            return warped
        
class Faiss_Engine():
    def __init__(self, opt):
        self.gallery_csv_path = opt.gallery_path
        self.faiss_index = None
        self.gallery_feas = None
        self.rec_threshold = opt.face_rec_threshold
        self.faiss_init()

    def faiss_init(self):
        # step1: read gallery to memory
        self.gallery_feas = []
        faiss_feas = []
        with open(self.gallery_csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.gallery_feas.append(row)
                faiss_feas.append(list(map(eval, row[1:])))
        # faiss add
        self.faiss_index = faiss.IndexFlatL2(len(faiss_feas[0]))
        faiss_feas_nparray = np.asarray(faiss_feas).astype(np.float32)
        self.faiss_index.add(faiss_feas_nparray)

    def face_search(self, features, img, boxes, landmarks):
        match_result = []
        for feature in features:
            # faiss搜索
            # print(feature.shape)
            feature_nparray = np.asarray([feature]).astype(np.float32)
            D, I = self.faiss_index.search(feature_nparray, 1)  # want to see k nearest neighbors
            # 取出当前index并转换成cos相似度
            feature_current = np.asarray(list(map(eval, self.gallery_feas[I[0][0]][1:]))).astype(np.float32)
            tmp_simi = np.dot(feature_current.reshape(1, 128)[0], feature_nparray[0])
            match_result.append([I[0][0], tmp_simi])
        gallery_ids = [i[0] for i in self.gallery_feas]
        pic = self.show_rec_result(img, boxes, landmarks, match_result, gallery_ids, self.rec_threshold)
        return pic

    def show_rec_result(self, img, boxes, landmarks, match_result, galler_ids, rec_threshold):
        if len(boxes) > 0:
            for i in range(boxes.shape[0]):
                box = boxes[i]
                color_text = (255, 255, 255)
                color_box = (255, 255, 0)
                color_landmark = (0, 255, 0)
                matched_id = galler_ids[match_result[i][0]].split('_')
                matched_simi = match_result[i][1]
                print("matched_id: " + str(matched_id[len(matched_id) - 1]) + " simi: " + str(matched_simi))
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_box, 2)
                if matched_simi > rec_threshold:
                    cv2.putText(img, str(matched_id[len(matched_id) - 1]) + ' ' + str(round(matched_simi * 100, 2)) + '%', (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_PLAIN, 2, color_text)
                if landmarks is not None:
                    landmark5 = landmarks[i]
                    for l in range(landmark5.shape[0]):
                        cv2.circle(img, (int(landmark5[l][0]), int(landmark5[l][1])), 1, color_landmark, 2)
            return img
        else:
            return img
