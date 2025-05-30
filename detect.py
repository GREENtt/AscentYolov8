#coding=utf-8

import numpy as np
import cv2
import os
import acl
from acllite_imageproc import AclLiteImage, AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource


input_imgH = 640
input_imgW = 640
conf_thres=0.25
iou_thres=0.45

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "./model/yolov8.om")
images_path = os.path.join(current_dir, "./data")
video_inference = False
video_path = "./E20241202110717_20241202110729.mp4"
output_path = './out'


# �Ǽ���ֵ���ƣ�NMS��
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

# ����
def postprocess(outputs, img_shape, conf_thres=0.25, iou_thres=0.45):
    outputs = np.transpose(np.squeeze(outputs))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    x_factor = img_shape[1] / input_imgW
    y_factor = img_shape[0] / input_imgH
    
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= conf_thres:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            x = x * x_factor
            y = y * y_factor
            w = w * x_factor
            h = h * y_factor
            boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            scores.append(max_score)
            class_ids.append(class_id)
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = nms(boxes, scores, iou_thres)
    
    return boxes[indices], scores[indices], np.array(class_ids)[indices]

# ͼƬԤ����
def preprocess_image(image, input_shape=(input_imgW, input_imgH)):
    original_shape = image.shape[:2]
    resized_img = cv2.resize(image, input_shape, interpolation=cv2.INTER_LINEAR)
    normalized_img = resized_img.astype(np.float32) / 255.0
    chw_img = np.transpose(normalized_img, (2, 0, 1))
    input_data = np.expand_dims(chw_img, axis=0)
    return input_data, original_shape

# ���Ƽ����
def draw_result(image, boxes, scores, class_ids):
    if len(boxes) == 0:
        return image
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        # ���Ʊ߽��
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # ������ǩ�ı�
        label = f"{classes[class_id]}:{score:.2f}"
        # �����ı��ߴ�
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # �����ı�����
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
        # �����ı�
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # ��ӡ�����
        print(f"Detect: {classes[class_id]} - Confidence: {score:.4f} - Location: [{x1}, {y1}, {x2}, {y2}]")
    return image

# ������
if __name__ == "__main__":
    # ��ʼ��ACLite��Դ
    acl_resource = AclLiteResource()
    acl_resource.init()

    # ��ʼ��ģ��
    model = AclLiteModel(model_path) 
    if model is None:
        print("Error: Model loading failed!")
        exit(1)
    print("Model loaded successfully.")
    
    # ȷ�����Ŀ¼����
    os.makedirs(output_path, exist_ok=True)
    
    if not video_inference:
        print('--> Processing images-----------------------------------------') 
        if not os.path.exists(images_path):
            raise Exception("The images path does not exist.")
        
        # ��������ͼ��
        for img_name  in os.listdir(images_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(images_path, img_name)
            print(f"Processing image: {img_name}")
            # ��ȡͼƬ
            image = cv2.imread(img_path)            
            if image is None:
                print(f"Failed read: {img_path}")
                continue
            
            # Ԥ����
            input_data, original_shape = preprocess_image(image)
            
            # ����
            output = model.execute([input_data])
            if output is None:
                print("Error: Inference failed!")
                exit(1)
            print("Inference succeeded.")
        
            # ����
            boxes, scores, class_ids = postprocess(output[0], original_shape)
            # ���ƽ��
            result_image = draw_result(image, boxes, scores, class_ids)
        
            # ������
            output_file = os.path.join(output_path,  f"det_{img_name}")
            cv2.imwrite(output_file, result_image)
            print(f"Save to: {output_file}\n")
            
    # ����Ƶ�ļ�
    else:
        print('--> Processing video -----------------------------------------')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")
            
        # ��ȡ��Ƶ����
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # ���������Ƶ
        output_path = os.path.join(output_dir, os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ԥ����
            input_data, original_shape = preprocess_image(frame)
            
            # ����
            output = model.execute([input_data])
            if output is None:
                print(f"Inference failed for frame {frame_count}")
            else:
                # ����
                boxes, scores, class_ids = postprocess(output[0], original_shape)
                # ���ƽ��
                frame = draw_result(frame, boxes, scores, class_ids)
            
            # д�������Ƶ
            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        # �ͷ���Դ
        cap.release()
        out.release()
        print(f"Video processing completed. Total frames: {frame_count}\n")
    
    # �ͷ���Դ
    model.destroy()
    acl.finalize()
    print("!!!!!!!!!!!!!!!!!!!!!!!SUCCESS!!!!!!!!!!!!!!!!!!!!!")
   