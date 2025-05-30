#coding=utf-8

import os
import cv2
import numpy as np
import acl
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

# 输入图像尺寸
input_imgH = 640
input_imgW = 640
# 置信度阈值和IoU阈值
conf_threshold = 0.25
iou_threshold = 0.45

# 类别名称
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

# 当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 模型路径
model_path = os.path.join(current_dir, "./model/yolov8n-seg.om")
# 图像路径
images_path = os.path.join(current_dir, "./data")
# 是否进行视频推理
video_inference = False
# 视频路径
video_path = "./E20241202110717_20241202110729.mp4"
# 输出路径
output_path = "./out"

# 颜色映射表
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)


# 图像预处理函数
def preprocess_image(image):
    # 获取原始图像尺寸
    h, w = image.shape[:2]
    # 计算缩放比例
    scale = min(input_imgH / h, input_imgW / w)
    # 计算新尺寸
    new_h, new_w = int(h * scale), int(w * scale)
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 创建填充后的图像
    padded = np.full((input_imgH, input_imgW, 3), 114, dtype=np.uint8)
    # 计算填充位置
    top = (input_imgH - new_h) // 2
    left = (input_imgW - new_w) // 2
    # 将缩放后的图像放入填充图像中
    padded[top:top+new_h, left:left+new_w] = resized
    # 转换为RGB
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # 归一化并转置维度
    input_data = padded_rgb.astype(np.float32) / 255.0
    input_data = input_data.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 640, 640)
    return input_data, (scale, top, left), image

# 非极大值抑制函数
def nms(boxes, scores, iou_threshold):
    # 按置信度降序排序
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        # 取当前最高置信度的框
        i = order[0]
        keep.append(i)
        
        # 计算当前框与其他框的IoU
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
        
        # 保留IoU低于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)

# 绘图函数：在原始图像上绘制检测框、标签和掩码
def draw_result(orig_img, boxes, class_ids, scores, masks):
    # 创建原始图像的副本
    result_img = orig_img.copy()
    # 创建彩色掩码图像
    mask_color = np.zeros_like(orig_img)
    
    # 处理每个检测结果
    for i in range(len(boxes)):
        class_id = int(class_ids[i])
        score = scores[i]
        box = boxes[i].astype(int)
        mask = masks[i]
        
        # 获取类别颜色
        color = [255,0,0] #colors[class_id].tolist()
        
        # 绘制边界框
        cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # 创建标签文本
        label = f"{classes[class_id]}:{score:.2f}"
        # 计算文本尺寸
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # 绘制文本背景
        cv2.rectangle(result_img, (box[0], box[1] - text_height - 10), (box[0] + text_width, box[1]), (0, 0, 255), -1)

        # 绘制文本
        cv2.putText(result_img, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 将掩码添加到彩色掩码图像
        mask_color[mask > 0] = color
        
        # 打印检测结果
        print(f"Detect: {classes[class_id]} - Confidence: {score:.4f} - Location: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
    
    # 将掩码以半透明方式叠加到原始图像
    result_img = cv2.addWeighted(result_img, 1, mask_color, 0.3, 0)
    
    return result_img

# 后处理函数
def postprocess(det_output, proto_output, scale, top, left, orig_img):
    # 获取原始图像尺寸
    orig_h, orig_w = orig_img.shape[:2]
    
    # 处理检测输出 (1, 116, 8400)
    det_output = det_output[0]  # (116, 8400)
    # 分离边界框、类别分数和掩码系数
    bbox_data = det_output[:4, :]  # (4, 8400)
    scores_data = det_output[4:4+80, :]  # (80, 8400)
    mask_coeff = det_output[4+80:, :]  # (32, 8400)
    
    # 转置以便处理
    bbox_data = bbox_data.T  # (8400, 4)
    scores_data = scores_data.T  # (8400, 80)
    mask_coeff = mask_coeff.T  # (8400, 32)
    
    # 获取每个框的最大类别分数
    class_ids = np.argmax(scores_data, axis=1)
    max_scores = scores_data[np.arange(len(scores_data)), class_ids]
    
    # 应用置信度阈值过滤
    valid_mask = max_scores > conf_threshold
    bbox_data = bbox_data[valid_mask]
    scores_data = scores_data[valid_mask]
    class_ids = class_ids[valid_mask]
    max_scores = max_scores[valid_mask]
    mask_coeff = mask_coeff[valid_mask]
    
    # 转换边界框格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
    cx, cy, w, h = bbox_data[:, 0], bbox_data[:, 1], bbox_data[:, 2], bbox_data[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # 执行非极大值抑制
    keep = nms(boxes, max_scores, iou_threshold)
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    max_scores = max_scores[keep]
    mask_coeff = mask_coeff[keep]
    
    # 处理分割输出 (1, 32, 160, 160)
    proto = proto_output[0]  # (32, 160, 160)
    c, mh, mw = proto.shape
    proto = proto.reshape(c, -1)  # (32, 25600)
    
    # 将边界框映射回原始图像
    boxes[:, 0] = (boxes[:, 0] - left) / scale  # x1
    boxes[:, 1] = (boxes[:, 1] - top) / scale  # y1
    boxes[:, 2] = (boxes[:, 2] - left) / scale  # x2
    boxes[:, 3] = (boxes[:, 3] - top) / scale  # y2
    
    # 确保边界框在图像范围内
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
    
    # 创建掩码列表
    masks = []
    
    # 处理每个检测结果
    for i in range(len(keep)):
        coeff = mask_coeff[i]
        
        # 计算掩码
        mask = np.matmul(coeff, proto).reshape(mh, mw)
        mask = 1 / (1 + np.exp(-mask))  # sigmoid激活
        
        # 阈值处理
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 将掩码缩放到原始图像尺寸
        mask = cv2.resize(mask, (input_imgW, input_imgH), interpolation=cv2.INTER_NEAREST)
        
        # 裁剪掩码的有效区域
        mask_roi = mask[top:top+int(orig_h*scale), left:left+int(orig_w*scale)]
        
        # 将掩码缩放到原始尺寸
        mask_orig = cv2.resize(mask_roi, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_orig)
    
    # 返回处理后的结果
    return boxes, class_ids, max_scores, masks

# 主函数
if __name__ == "__main__":
    acl_resource = AclLiteResource()
    acl_resource.init()
    
    model = AclLiteModel(model_path)
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 视频推理
    if video_inference:
        print('--> Processing video-----------------------------------------')
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 预处理
            input_data, (scale, top, left), orig_img = preprocess_image(frame)
            
            # 推理
            det_out, proto_out = model.execute(input_data)
            
            # 后处理
            boxes, class_ids, scores, masks = postprocess(det_out, proto_out, scale, top, left, frame.copy())
            
            # 绘制结果
            result_img = draw_result(frame.copy(), boxes, class_ids, scores, masks)
            
            # 保存结果
            output_file = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_file, result_img)
            print(f"Save to: {output_file}\n")            
            frame_count += 1
        
        cap.release()
    
    # 图像推理
    else:
        # 遍历所有图像
        print('--> Processing images-----------------------------------------')
        for img_name in os.listdir(images_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(images_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed read: {img_path}")
                continue
            
            # 预处理
            input_data, (scale, top, left), orig_img = preprocess_image(img)
            
            # 推理
            det_out, proto_out = model.execute(input_data)
            
            # 后处理
            boxes, class_ids, scores, masks = postprocess(det_out, proto_out, scale, top, left, img.copy())
            
            # 绘制结果
            result_img = draw_result(img.copy(), boxes, class_ids, scores, masks)
            
            # 保存结果
            output_file = os.path.join(output_path, f"seg_{img_name}")
            cv2.imwrite(output_file, result_img)
            print(f"Save to: {output_file}\n")
    
    # 释放资源
    model.destroy()
    acl.finalize()
    print("!!!!!!!!!!!!!!!!!!!!!!!SUCCESS!!!!!!!!!!!!!!!!!!!!!")
