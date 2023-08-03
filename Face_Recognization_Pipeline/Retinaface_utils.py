import cv2
import numpy as np
import tensorflow as tf
from math import ceil
from itertools import product as product
import time


def letterbox_image(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh))
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image

# retinaface decode
def retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1],
                           scale[0]]

    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0],
                            offset[1], offset[0]]

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result


class BoxUtil(object):
    def __init__(self, top_k=300, nms_thresh=0.45):

        self._top_k = top_k
        self._nms_thresh = nms_thresh

    def detection_out_aid(self, boxes_aid, scores_aid, landmarks_aid, anchorbox, confidence_threshold=0.7):
        # boxes
        pred_loc = boxes_aid
        # confidence
        pred_conf = scores_aid[:, 1:2]
        # landmarks
        pred_ldm = landmarks_aid
        # decode_bbox
        decode_bbox = self.decode_boxes(pred_loc, pred_ldm, anchorbox)
        # 置信度过滤
        conf_mask = (pred_conf >= confidence_threshold)[:, 0]
        # boxes, scores, landmarks合并
        detection = np.concatenate((decode_bbox[conf_mask][:, :4], pred_conf[conf_mask], decode_bbox[conf_mask][:, 4:]), -1)
        # nms
        idx = tf.image.non_max_suppression(tf.cast(detection[:, :4], tf.float32), tf.cast(detection[:, 4], tf.float32), self._top_k, iou_threshold=self._nms_thresh).numpy()
        true_box = detection[idx]
        return true_box

    def decode_boxes(self, pred_loc, pred_ldm, anchorbox):
        # Anchor宽高
        anchor_width = anchorbox[:, 2] - anchorbox[:, 0]
        anchor_height = anchorbox[:, 3] - anchorbox[:, 1]
        # Anchor中心点
        anchor_center_x = 0.5 * (anchorbox[:, 2] + anchorbox[:, 0])
        anchor_center_y = 0.5 * (anchorbox[:, 3] + anchorbox[:, 1])
        # predict框距离Anchor中心的xy轴偏移情况
        decode_bbox_center_x = pred_loc[:, 0] * anchor_width * 0.1
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = pred_loc[:, 1] * anchor_height * 0.1
        decode_bbox_center_y += anchor_center_y
        # predict框宽高
        decode_bbox_width = np.exp(pred_loc[:, 2] * 0.2)
        decode_bbox_width *= anchor_width
        decode_bbox_height = np.exp(pred_loc[:, 3] * 0.2)
        decode_bbox_height *= anchor_height
        # predict box
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        anchor_width = np.expand_dims(anchor_width, -1)
        anchor_height = np.expand_dims(anchor_height, -1)
        anchor_center_x = np.expand_dims(anchor_center_x, -1)
        anchor_center_y = np.expand_dims(anchor_center_y, -1)

        # Anchor的五个人脸关键点
        pred_ldm = pred_ldm.reshape([-1, 5, 2])
        decode_ldm = np.zeros_like(pred_ldm)
        decode_ldm[:, :, 0] = np.repeat(anchor_width, 5, axis=-1) * pred_ldm[:, :, 0] * 0.1 + np.repeat(anchor_center_x, 5, axis=-1)
        decode_ldm[:, :, 1] = np.repeat(anchor_height, 5, axis=-1) * pred_ldm[:, :, 1] * 0.1 + np.repeat(anchor_center_y, 5, axis=-1)
        # 真实框landmarks
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None], decode_bbox_ymin[:, None], decode_bbox_xmax[:, None], decode_bbox_ymax[:, None], np.reshape(decode_ldm, [-1, 10])), axis=-1)
        # decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox


class Anchors(object):
    def __init__(self, image_size=None):
        super(Anchors, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        # 图片的尺寸
        self.image_size = image_size
        # 三个有效特征层高和宽
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # 对特征层的高和宽进行循环迭代
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        anchors = np.reshape(anchors, [-1, 4])

        output = np.zeros_like(anchors[:, :4])
        # xywh2xyxy
        output[:, 0] = anchors[:, 0] - anchors[:, 2] / 2
        output[:, 1] = anchors[:, 1] - anchors[:, 3] / 2
        output[:, 2] = anchors[:, 0] + anchors[:, 2] / 2
        output[:, 3] = anchors[:, 1] + anchors[:, 3] / 2

        return output


class Retinaface():
    def __init__(self, opt):
        self.backbone = opt.retinaface_backbone
        self.confidence = opt.retinaface_threshold
        self.nms_iou = opt.retinaface_nms_iou
        self.input_shape = opt.retinaface_input_shape

        #aidlux
        self.retinaface_model_path = opt.retinaface_model_path
        self.retinaface_input_shape = [1 * opt.retinaface_input_shape[0] * opt.retinaface_input_shape[1] * opt.retinaface_input_shape[2] * 4]
        self.retinaface_output_shape = [1 * opt.retinaface_output_shape[0] * 16800 *4, 1 * opt.retinaface_output_shape[1] * 16800 *4, 1 * opt.retinaface_output_shape[2] * 16800 *4,]

        self.scale = None
        self.scale_for_landmarks = None

        self.bbox_util = BoxUtil(nms_thresh=self.nms_iou)
        self.anchors = Anchors(image_size=(self.input_shape[0], self.input_shape[1])).get_anchors()

    def pre_process(self, image):
        raw_image = image.copy()
        # 把图像转换成numpy
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        # 计算预测图的缩放比例
        self.scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        self.scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],np.shape(image)[1], np.shape(image)[0]]

        # letterbox resize
        image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        return image, raw_image, im_height, im_width

    def draw_retinaface_result(self, raw_image, results_aid):
        for b in results_aid:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(raw_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(raw_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.circle(raw_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(raw_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(raw_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(raw_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(raw_image, (b[13], b[14]), 1, (255, 0, 0), 4)
        return raw_image

    def post_process(self, raw_image, boxes_aid, scores_aid, landmarks_aid, im_height, im_width):
        results_aid = self.bbox_util.detection_out_aid(boxes_aid, scores_aid, landmarks_aid, self.anchors, confidence_threshold=self.confidence)

        # 没有预测结果的话直接返回原图
        if len(results_aid) <= 0:
            return raw_image, results_aid

        results_aid = retinaface_correct_boxes(results_aid, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))

        results_aid[:, :4] = results_aid[:, :4] * self.scale
        results_aid[:, 5:] = results_aid[:, 5:] * self.scale_for_landmarks

        retinaface_image = self.draw_retinaface_result(raw_image, results_aid)

        return retinaface_image, results_aid

    def detect_image(self, raw_frame, AidliteEngine):
        # retinaface preprocess
        time_aid1 = time.time()
        image, raw_image, im_height, im_width= self.pre_process(raw_frame)
        # print('retinaface preprocess time: {:.4f}'.format(time.time() - time_aid1))

        # aidlux invoke
        time_aid2 = time.time()
        img_aid = np.float32(image)
        AidliteEngine.set_g_index(0)
        AidliteEngine.setInput_Float32(img_aid, 640, 640)
        AidliteEngine.invoke()
        landmarks_aid = AidliteEngine.getOutput_Float32(0).reshape(1, 10, 16800)[0].transpose()
        scores_aid = AidliteEngine.getOutput_Float32(1).reshape(1, 2, 16800)[0].transpose()
        boxes_aid = AidliteEngine.getOutput_Float32(2).reshape(1, 4, 16800)[0].transpose()
        # print('aid retinaface forward time: {:.4f}'.format(time.time() - time_aid2))

        # retinaface postprocess
        time_aid3 = time.time()
        result_image, retinaface_pred = self.post_process(raw_image, boxes_aid, scores_aid, landmarks_aid, im_height, im_width)
        # print('retinaface postprocess time: {:.4f}'.format(time.time() - time_aid3))
        return result_image, retinaface_pred

    def detect_image_onnx(self, raw_frame, OnnxEngine):

        image, raw_image, im_height, im_width = self.pre_process(raw_frame)
        img = image.transpose(2, 0, 1).reshape(1, 3, 640, 640)
        img = img.astype(np.float32)
        preds = OnnxEngine.run(None, {'input0': img})
        boxes_aid = preds[0][0]
        scores_aid = preds[1][0]
        landmarks_aid = preds[2][0]

        result_image, retinaface_pred = self.post_process(raw_image, boxes_aid, scores_aid, landmarks_aid, im_height, im_width)
        # print('retinaface postprocess time: {:.4f}'.format(time.time() - time_aid3))
        return result_image, retinaface_pred