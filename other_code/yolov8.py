# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from acllite_imageproc import AclLiteImage, AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

# 示例：COCO 数据集的类别名称
labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

class sample_YOLOV8_NMS_ONNX(object):
    def __init__(self, yolo_model_path, yolo_model_width, yolo_model_height, nms_threshold=0.5, iou_threshold=0.5):
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        self.yolo_model_width = yolo_model_width
        self.yolo_model_height = yolo_model_height
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold
        self.yolo_result = None
        self.resource = None
        self.dvpp = None
        self.image = None
        self.resized_image = None

    def init_resource(self):
        # 初始化 ACL 资源
        self.resource = AclLiteResource()
        self.resource.init()
        # 初始化 DVPP 资源
        self.dvpp = AclLiteImageProc(self.resource)
        # 加载 YOLOv8 模型
        self.yolo_model = AclLiteModel(self.yolo_model_path)

    def yolo_process_input(self, input_path):
        # 读取图片
        self.image = AclLiteImage(input_path)
        # 将图片从主机内存复制到 DVPP
        image_input = self.image.copy_to_dvpp()
        # 解码图片
        yuv_image = self.dvpp.jpegd(image_input)
        # 调整图片大小
        self.resized_image = self.dvpp.crop_and_paste(yuv_image, self.image.width, self.image.height,
                                                      self.yolo_model_width, self.yolo_model_height)

    def yolo_inference(self):
        # 执行推理
        self.yolo_result = self.yolo_model.execute([self.resized_image])
        print(f"YOLOv8 Model Output Type: {type(self.yolo_result)}")
        print(f"YOLOv8 Model Output Shape: {self.yolo_result[0].shape}")
        print(f"YOLOv8 Model Output Sample: {self.yolo_result[0][:, :5, :5]}")  # 打印部分输出

    def decode_detections(self, outputs, conf_threshold, width, height):
        # 解码检测框
        detections = []
        for detection in outputs[0][0]:  # outputs[0][0] 是 [84, 8400] 的数组
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                if class_id >= len(labels):
                    print(f"Class ID {class_id} is out of range for labels list. Skipping this detection.")
                    continue
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detections.append([x, y, w, h, confidence, class_id])
        print(f"Number of detections after decoding: {len(detections)}")
        return detections

    def postprocess(self):
        # 解码模型输出
        conf_threshold = 0.2  # 调整置信度阈值
        detections = self.decode_detections(self.yolo_result, conf_threshold, self.yolo_model_width, self.yolo_model_height)

        # 应用 NMS
        boxes = []
        scores = []
        class_ids = []
        for detection in detections:
            x, y, w, h, confidence, class_id = detection
            boxes.append([x, y, x + w, y + h])
            scores.append(confidence)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.nms_threshold, self.iou_threshold)
        if len(indices) == 0:
            print("No valid detections after NMS.")
            return [], [], []

        print(f"Number of detections after NMS: {len(indices.flatten())}")
        return [boxes[i] for i in indices.flatten()], [scores[i] for i in indices.flatten()], [class_ids[i] for i in indices.flatten()]

    def draw_results(self, src_image_path, boxes, scores, class_ids):
        if not boxes or not scores or not class_ids:
            print("No valid detections to draw.")
            return

        src_image = cv2.imread(src_image_path)
        if src_image is None:
            print(f"Failed to load image: {src_image_path}")
            return

        scale_x = src_image.shape[1] / self.yolo_model_width
        scale_y = src_image.shape[0] / self.yolo_model_height
        max_scale = max(scale_x, scale_y)
        colors = [0, 0, 255]

        print(f"Length of boxes: {len(boxes)}")
        print(f"Length of scores: {len(scores)}")
        print(f"Length of class_ids: {len(class_ids)}")

        for box, score, class_id in zip(boxes, scores, class_ids):
            if class_id >= len(labels):
                print(f"Class ID {class_id} is out of range for labels list. Skipping this detection.")
                continue

            label = f"{labels[class_id]}:{score:.2f}"
            top_left_x = box[0] * max_scale
            top_left_y = box[1] * max_scale
            bottom_right_x = box[2] * max_scale
            bottom_right_y = box[3] * max_scale

            if top_left_x < 0 or top_left_y < 0 or bottom_right_x > src_image.shape[1] or bottom_right_y > src_image.shape[0]:
                print(f"Invalid box coordinates: {box}")
                continue

            cv2.rectangle(src_image, (int(top_left_x), int(top_left_y)),
                          (int(bottom_right_x), int(bottom_right_y)), colors)
            p3 = (max(int(top_left_x), 15), max(int(top_left_y), 15))
            cv2.putText(src_image, label, p3, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors, 1)

        output_path = os.path.join("./out", os.path.basename(src_image_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, src_image)

    def release_resource(self):
        # 释放资源
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
        raise Exception("The YOLOv8 model path does not exist.")
    if not os.path.exists(images_path):
        raise Exception("The images path does not exist.")
    all_path = [os.path.join(images_path, path) for path in os.listdir(images_path) if path != '.keep']
    if len(all_path) == 0:
        raise Exception("The directory is empty, please download images.")

    net = sample_YOLOV8_NMS_ONNX(yolo_model_path, yolo_width, yolo_height, nms_threshold=0.5, iou_threshold=0.5)
    net.init_resource()
    for image_path in all_path:
        net.yolo_process_input(image_path)
        net.yolo_inference()
        boxes, scores, class_ids = net.postprocess()
        net.draw_results(image_path, boxes, scores, class_ids)
    net.release_resource()
    print("Success")