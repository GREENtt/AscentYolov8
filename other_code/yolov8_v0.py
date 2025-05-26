# -*- coding: utf-8

"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2024-05-25 09:12:13
MODIFIED: 2024-05-25 10:10:55
"""
import os
import numpy as np
import cv2
from acllite_imageproc import AclLiteImage
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
from acllite_logger import log_info
from label import labels  # �����ǩ�ļ���YOLOv7һ�£��ɸ���YOLOv8ʵ���������


class sample_YOLOV8_NMS_ONNX(object):
    def __init__(self, yolo_model_path, yolo_model_width, yolo_model_height):
        self.yolo_model_path = yolo_model_path    # string
        self.yolo_model = None
        self.yolo_model_width = yolo_model_width
        self.yolo_model_height = yolo_model_height
        self.yolo_result = None
        self.resource = None
        self.dvpp = None
        self.image = None
        self.resized_image = None

    def init_resource(self):
        # init acl resource
        self.resource = AclLiteResource()
        self.resource.init()
        # init dvpp resource
        self.dvpp = AclLiteImageProc(self.resource)
        # load yolo model from file
        self.yolo_model = AclLiteModel(self.yolo_model_path)

    def yolo_process_input(self, input_path):
        # read image from file
        self.image = AclLiteImage(input_path)
        # memory copy from host to dvpp
        image_input = self.image.copy_to_dvpp()
        # decode image from JPEGD format to YUV
        yuv_image = self.dvpp.jpegd(image_input)
        # execute resize
        self.resized_image = self.dvpp.crop_and_paste(yuv_image, self.image.width, self.image.height,
                                                      self.yolo_model_width, self.yolo_model_height)

    def yolo_inference(self):
        # inference
        self.yolo_result = self.yolo_model.execute([self.resized_image])

    def yolo_get_result(self, src_image_path, confidence_threshold=0.5):
        # YOLOv8 �����������߼���YOLOv7������ͬ���������YOLOv8�������ʽ���н���
        # �������Ϊһ����������������������Ŷȵ���Ϣ
        # �����ʽ�����ʵ��ģ���������
        outputs = self.yolo_result[0]  # ��ȡģ�����
        boxes = outputs[:, :4]  # ��������
        scores = outputs[:, 4]  # ���Ŷ�
        class_ids = outputs[:, 5].astype(int)  # ���ID
        
        # ��� class_ids �� scores ����״
        if class_ids.ndim > 1:
            class_ids = class_ids.flatten()  # ����ά����չƽΪһά����
        if scores.ndim > 1:
            scores = scores.flatten()  # ����ά����չƽΪһά����
        # ��� boxes ����״
        if boxes.ndim > 2:
            boxes = boxes.reshape(-1, 4)  # ȷ�� boxes �� (N, 4) ��״

        src_image = cv2.imread(src_image_path)
        scale_x = src_image.shape[1] / self.yolo_model_width
        scale_y = src_image.shape[0] / self.yolo_model_height
        # get scale factor
        max_scale = max(scale_x, scale_y)
        colors = [0, 0, 255]
        
        # draw the boxes in original image
        for i in range(len(boxes)):
            # ȷ�� class_ids[i] �� scores[i] �Ǳ���
            class_id = class_ids[i].item() if isinstance(class_ids[i], np.ndarray) else class_ids[i]
            score = scores[i].item() if isinstance(scores[i], np.ndarray) else scores[i]
            # ���˵����Ŷȵļ���
            if score < confidence_threshold:
                continue
            label = labels[class_ids[i]] + ":" + str("%.2f" % scores[i])
            top_left_x = boxes[i][0] * max_scale
            top_left_y = boxes[i][1] * max_scale
            bottom_right_x = boxes[i][2] * max_scale
            bottom_right_y = boxes[i][3] * max_scale
            cv2.rectangle(src_image, (int(top_left_x), int(top_left_y)),
                          (int(bottom_right_x), int(bottom_right_y)), colors)
            p3 = (max(int(top_left_x), 15), max(int(top_left_y), 15))
            cv2.putText(src_image, label, p3, cv2.FONT_ITALIC, 0.6, colors, 1)
        output_path = os.path.join("./out", os.path.basename(src_image_path))
        cv2.imwrite(output_path, src_image)

    def release_resource(self):
        # release resource includes acl resource, data set and unload model
        self.dvpp.__del__()
        self.yolo_model.__del__()
        self.resource.__del__()
        AclLiteResource.__del__ = lambda x: 0
        AclLiteImage.__del__ = lambda x: 0
        AclLiteImageProc.__del__ = lambda x: 0
        AclLiteModel.__del__ = lambda x: 0

if __name__ == "__main__":
    yolo_width = 640
    yolo_height = 640
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_model_path = os.path.join(current_dir, "./model/yolov8n.om")
    images_path = os.path.join(current_dir, "./data")
    if not os.path.exists(yolo_model_path):
        raise Exception("the yolo model path is not exist")
    if not os.path.exists(images_path):
        raise Exception("the images path is not exist")
    all_path = []
    for path in os.listdir(images_path):
        if path != '.keep':
            total_path = os.path.join(images_path, path)
            all_path.append(total_path)
    if len(all_path) == 0:
        raise Exception("the directory is empty, please download image")
    net = sample_YOLOV8_NMS_ONNX(yolo_model_path, yolo_width, yolo_height)
    net.init_resource()
    for images_path in all_path:
        net.yolo_process_input(images_path)
        net.yolo_inference()
        net.yolo_get_result(images_path, confidence_threshold=0.5)
    net.release_resource()
    log_info("success")
