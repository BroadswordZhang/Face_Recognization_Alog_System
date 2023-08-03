import sys

import cv2
import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


class CenterFaceEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cali_dir, cache_file, batch_size=1):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        super(CenterFaceEntropyCalibrator, self).__init__()

        self.all_files = []
        for root, dirs, files in os.walk(cali_dir):
            for file in files:
                if os.path.splitext(file)[1] in ['.jpg', '.png']:
                    self.all_files.append(os.path.join(root, file))

        self.batch_size = batch_size
        self.current_index = 0
        self.cache_file = cache_file
        self.whole_len = len(self.all_files)
        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * 1920 * 1080 * 4)

    def get_batch_size(self):
        return self.batch_size

    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.whole_len:
            print("all calibrated self.current_index + self.batch_size > self.whole_len \n".format(
                self.current_index, self.batch_size, self.whole_len))
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 1 == 0:
            print("Calibrating batch {:}, containing {:} images, whole:{}".format(current_batch, self.batch_size,
                                                                                  len(self.all_files)))

        # batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        batch = None
        for i in range(self.current_index, self.current_index + self.batch_size):
            img = cv2.imread(self.all_files[self.current_index])

            # # should be same with optimized profile while engine building.
            img = cv2.resize(img, (544, 960))
            img_h_new, img_w_new, scale_h, scale_w = self.transform(img.shape[0], img.shape[1])
            one_node = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(img_w_new, img_h_new), mean=(0, 0, 0),
                                             swapRB=True, crop=False)
            if batch is None:
                batch = one_node
            else:
                batch = np.concatenate((batch, one_node), 0)
        # print("batch {}".format(self.current_index))
        sys.stdout.flush()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
