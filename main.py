#coding=utf-8

import numpy as np
import cv2
import os
import acl
from acllite_imageproc import AclLiteImage, AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource


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

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "./model/yolov8.om")
images_path = os.path.join(current_dir, "./data")



# 非极大值抑制（NMS）
def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep

# 后处理
def postprocess(outputs, img_shape, conf_thres=0.25, iou_thres=0.45):
    print("Postprocessing output shape:", outputs.shape)
    outputs = np.transpose(np.squeeze(outputs))  # 转置并压缩输出
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    x_factor = img_shape[1] / 640  # 假设模型输入尺寸为640x640
    y_factor = img_shape[0] / 640
    for i in range(rows):
        classes_scores = outputs[i][4:]  # 提取类别分数
        max_score = np.amax(classes_scores)
        if max_score >= conf_thres:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            x = x * x_factor
            y = y * y_factor
            w = w * x_factor
            h = h * y_factor
            boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])  # 转换为左上角和右下角坐标
            scores.append(max_score)
            class_ids.append(class_id)
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = nms(boxes, scores, iou_thres)
    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = np.array(class_ids)[indices]
    return boxes, scores, class_ids

# 图片预处理
def preprocess_image(image_path, input_shape=(640, 640)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    original_shape = image.shape[:2]
    image = cv2.resize(image, input_shape, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0  # 归一化
    image = np.transpose(image, (2, 0, 1))  # 转换为CHW格式
    image = np.expand_dims(image, axis=0)  # 添加batch维度
    print(f"Input image shape: {image.shape}, dtype: {image.dtype}")
    return image, original_shape

# 主函数
if __name__ == "__main__":
    # 初始化ACLite资源
    acl_resource = AclLiteResource()
    acl_resource.init()

    # 初始化模型
    #model_path = "./model/yolov8.om"  # ACLite模型路径
    model = AclLiteModel(model_path)
    if model is None:
        print("Error: Model loading failed!")
        exit(1)
    else:
        print("Model loaded successfully.")

    # 图片预处理  
    if not os.path.exists(images_path):
        raise Exception("The images path does not exist.")
    all_path = [os.path.join(images_path, path) for path in os.listdir(images_path) if path != '.keep']
    if len(all_path) == 0:
        raise Exception("The directory is empty, please download images.")
    #image_path = "./data/test.jpg"  # 输入图片路径
    for image_path in all_path:
        input_image, original_shape = preprocess_image(image_path)
        print(f"Input image shape: {input_image.shape}")
    
        # 执行推理
        output = model.execute([input_image])
        if output is None:
            print("Error: Inference failed!")
            exit(1)
        else:
            print("Inference succeeded.")
    
        # 后处理
        boxes, scores, class_ids = postprocess(output[0], original_shape)
    
        # 可视化结果
        image = cv2.imread(image_path)
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"{labels[class_id]}:{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        output_path = os.path.join("./out", os.path.basename(image_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

    # 释放资源
    acl.finalize()
    print("Success")