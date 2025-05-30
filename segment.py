#coding=utf-8

import os
import cv2
import numpy as np
import acl
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

# ����ͼ��ߴ�
input_imgH = 640
input_imgW = 640
# ���Ŷ���ֵ��IoU��ֵ
conf_threshold = 0.25
iou_threshold = 0.45

# �������
classes = [
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

# ��ǰ�ű�Ŀ¼
current_dir = os.path.dirname(os.path.abspath(__file__))
# ģ��·��
model_path = os.path.join(current_dir, "./model/yolov8n-seg.om")
# ͼ��·��
images_path = os.path.join(current_dir, "./data")
# �Ƿ������Ƶ����
video_inference = False
# ��Ƶ·��
video_path = "./E20241202110717_20241202110729.mp4"
# ���·��
output_path = "./out"

# ��ɫӳ���
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)


# ͼ��Ԥ������
def preprocess_image(image):
    # ��ȡԭʼͼ��ߴ�
    h, w = image.shape[:2]
    # �������ű���
    scale = min(input_imgH / h, input_imgW / w)
    # �����³ߴ�
    new_h, new_w = int(h * scale), int(w * scale)
    # ����ͼ��
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # ���������ͼ��
    padded = np.full((input_imgH, input_imgW, 3), 114, dtype=np.uint8)
    # �������λ��
    top = (input_imgH - new_h) // 2
    left = (input_imgW - new_w) // 2
    # �����ź��ͼ��������ͼ����
    padded[top:top+new_h, left:left+new_w] = resized
    # ת��ΪRGB
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # ��һ����ת��ά��
    input_data = padded_rgb.astype(np.float32) / 255.0
    input_data = input_data.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 640, 640)
    return input_data, (scale, top, left), image

# �Ǽ���ֵ���ƺ���
def nms(boxes, scores, iou_threshold):
    # �����ŶȽ�������
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        # ȡ��ǰ������ŶȵĿ�
        i = order[0]
        keep.append(i)
        
        # ���㵱ǰ�����������IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_others - intersection
        
        iou = intersection / (union + 1e-7)
        
        # ����IoU������ֵ�Ŀ�
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)

# ��ͼ��������ԭʼͼ���ϻ��Ƽ��򡢱�ǩ������
def draw_result(orig_img, boxes, class_ids, scores, masks):
    # ����ԭʼͼ��ĸ���
    result_img = orig_img.copy()
    # ������ɫ����ͼ��
    mask_color = np.zeros_like(orig_img)
    
    # ����ÿ�������
    for i in range(len(boxes)):
        class_id = int(class_ids[i])
        score = scores[i]
        box = boxes[i].astype(int)
        mask = masks[i]
        
        # ��ȡ�����ɫ
        color = [255,0,0] #colors[class_id].tolist()
        
        # ���Ʊ߽��
        cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # ������ǩ�ı�
        label = f"{classes[class_id]}:{score:.2f}"
        # �����ı��ߴ�
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # �����ı�����
        cv2.rectangle(result_img, (box[0], box[1] - text_height - 10), (box[0] + text_width, box[1]), (0, 0, 255), -1)

        # �����ı�
        cv2.putText(result_img, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ��������ӵ���ɫ����ͼ��
        mask_color[mask > 0] = color
        
        # ��ӡ�����
        print(f"Detect: {classes[class_id]} - Confidence: {score:.4f} - Location: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    
    # �������԰�͸����ʽ���ӵ�ԭʼͼ��
    result_img = cv2.addWeighted(result_img, 1, mask_color, 0.3, 0)
    
    return result_img

# ������
def postprocess(det_output, proto_output, scale, top, left, orig_img):
    # ��ȡԭʼͼ��ߴ�
    orig_h, orig_w = orig_img.shape[:2]
    
    # ��������� (1, 116, 8400)
    det_output = det_output[0]  # (116, 8400)
    # ����߽��������������ϵ��
    bbox_data = det_output[:4, :]  # (4, 8400)
    scores_data = det_output[4:4+80, :]  # (80, 8400)
    mask_coeff = det_output[4+80:, :]  # (32, 8400)
    
    # ת���Ա㴦��
    bbox_data = bbox_data.T  # (8400, 4)
    scores_data = scores_data.T  # (8400, 80)
    mask_coeff = mask_coeff.T  # (8400, 32)
    
    # ��ȡÿ��������������
    class_ids = np.argmax(scores_data, axis=1)
    max_scores = scores_data[np.arange(len(scores_data)), class_ids]
    
    # Ӧ�����Ŷ���ֵ����
    valid_mask = max_scores > conf_threshold
    bbox_data = bbox_data[valid_mask]
    scores_data = scores_data[valid_mask]
    class_ids = class_ids[valid_mask]
    max_scores = max_scores[valid_mask]
    mask_coeff = mask_coeff[valid_mask]
    
    # ת���߽���ʽ (cx, cy, w, h) -> (x1, y1, x2, y2)
    cx, cy, w, h = bbox_data[:, 0], bbox_data[:, 1], bbox_data[:, 2], bbox_data[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # ִ�зǼ���ֵ����
    keep = nms(boxes, max_scores, iou_threshold)
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    max_scores = max_scores[keep]
    mask_coeff = mask_coeff[keep]
    
    # ����ָ���� (1, 32, 160, 160)
    proto = proto_output[0]  # (32, 160, 160)
    c, mh, mw = proto.shape
    proto = proto.reshape(c, -1)  # (32, 25600)
    
    # ���߽��ӳ���ԭʼͼ��
    boxes[:, 0] = (boxes[:, 0] - left) / scale  # x1
    boxes[:, 1] = (boxes[:, 1] - top) / scale  # y1
    boxes[:, 2] = (boxes[:, 2] - left) / scale  # x2
    boxes[:, 3] = (boxes[:, 3] - top) / scale  # y2
    
    # ȷ���߽����ͼ��Χ��
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
    
    # ���������б�
    masks = []
    
    # ����ÿ�������
    for i in range(len(keep)):
        coeff = mask_coeff[i]
        
        # ��������
        mask = np.matmul(coeff, proto).reshape(mh, mw)
        mask = 1 / (1 + np.exp(-mask))  # sigmoid����
        
        # ��ֵ����
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # ���������ŵ�ԭʼͼ��ߴ�
        mask = cv2.resize(mask, (input_imgW, input_imgH), interpolation=cv2.INTER_NEAREST)
        
        # �ü��������Ч����
        mask_roi = mask[top:top+int(orig_h*scale), left:left+int(orig_w*scale)]
        
        # ���������ŵ�ԭʼ�ߴ�
        mask_orig = cv2.resize(mask_roi, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_orig)
    
    # ���ش����Ľ��
    return boxes, class_ids, max_scores, masks

# ������
if __name__ == "__main__":
    acl_resource = AclLiteResource()
    acl_resource.init()
    
    model = AclLiteModel(model_path)
    # �������Ŀ¼
    os.makedirs(output_path, exist_ok=True)
    
    # ��Ƶ����
    if video_inference:
        print('--> Processing video-----------------------------------------')
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ԥ����
            input_data, (scale, top, left), orig_img = preprocess_image(frame)
            
            # ����
            det_out, proto_out = model.execute(input_data)
            
            # ����
            boxes, class_ids, scores, masks = postprocess(det_out, proto_out, scale, top, left, frame.copy())
            
            # ���ƽ��
            result_img = draw_result(frame.copy(), boxes, class_ids, scores, masks)
            
            # ������
            output_file = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_file, result_img)
            print(f"Save to: {output_file}\n")            
            frame_count += 1
        
        cap.release()
    
    # ͼ������
    else:
        # ��������ͼ��
        print('--> Processing images-----------------------------------------')
        for img_name in os.listdir(images_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(images_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed read: {img_path}")
                continue
            
            # Ԥ����
            input_data, (scale, top, left), orig_img = preprocess_image(img)
            
            # ����
            det_out, proto_out = model.execute(input_data)
            
            # ����
            boxes, class_ids, scores, masks = postprocess(det_out, proto_out, scale, top, left, img.copy())
            
            # ���ƽ��
            result_img = draw_result(img.copy(), boxes, class_ids, scores, masks)
            
            # ������
            output_file = os.path.join(output_path, f"seg_{img_name}")
            cv2.imwrite(output_file, result_img)
            print(f"Save to: {output_file}\n")
    
    # �ͷ���Դ
    model.destroy()
    acl.finalize()
    print("!!!!!!!!!!!!!!!!!!!!!!!SUCCESS!!!!!!!!!!!!!!!!!!!!!")
