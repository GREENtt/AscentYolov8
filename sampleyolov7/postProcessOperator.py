# -*- coding: utf-8

"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2023-05-25 09:12:13
MODIFIED: 2023-05-25 10:10:55
"""
import onnx
from onnx import helper

# YOLOv8n相关参数
h, w = 640, 640
boxNum = 3
outNUm = 3
classes = 80
coords = 4
f_h, f_w = h // 8, w // 8
# yolov8n anchor
anchor = [10.0, 13.0,
16.0, 30.0,
33.0, 23.0,
30.0, 61.0,
62.0, 45.0,
59.0, 119.0,
116.0, 90.0,
156.0, 198.0,
373.0, 326.0]
input_shape_0 = [1, 255, 80, 80]
input_shape_1 = [1, 255, 40, 40]
input_shape_2 = [1, 255, 20, 20]
img_info_num = 4
max_boxes_out = 6 * 1024
box_num = 8
pre_nms_topn = 1024
post_nms_topn = 1024
relative = 1
out_box_dim = 2
obj_threshold = 0.25
score_threshold = 0.25
iou_threshold = 0.45

# 定义输入
input_0 = helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, input_shape_0)
input_1 = helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, input_shape_1)
input_2 = helper.make_tensor_value_info("input_2", onnx.TensorProto.FLOAT, input_shape_2)

# YOLOv8n预处理节点
crd_align_len = f_h * f_w
obj_align_len = boxNum * f_h * f_w
coord_data_0 = helper.make_tensor_value_info("yolo_coord_0", onnx.TensorProto.FLOAT,
['batch', boxNum * 4, crd_align_len])
obj_prob_0 = helper.make_tensor_value_info("yolo_obj_0", onnx.TensorProto.FLOAT,
['batch', obj_align_len])
classes_prob_0 = helper.make_tensor_value_info("yolo_classes_0", onnx.TensorProto.FLOAT,
['batch', classes, obj_align_len])
yolo_pre_node_0 = helper.make_node('YoloPreDetection',
inputs = ["input_0"],
outputs = ["yolo_coord_0", "yolo_obj_0", "yolo_classes_0"],
boxes = boxNum,
coords = coords,
classes = classes,
yolo_version ='V8',
name = "yolo_pre_node_0")

f_h, f_w = f_h // 2, f_w // 2
crd_align_len = f_h * f_w
obj_align_len = boxNum * f_h * f_w
coord_data_1 = helper.make_tensor_value_info("yolo_coord_1", onnx.TensorProto.FLOAT,
['batch', boxNum * 4, crd_align_len])
obj_prob_1 = helper.make_tensor_value_info("yolo_obj_1", onnx.TensorProto.FLOAT,
['batch', obj_align_len])
classes_prob_1 = helper.make_tensor_value_info("yolo_classes_1", onnx.TensorProto.FLOAT,
['batch', classes, obj_align_len])
yolo_pre_node_1 = helper.make_node('YoloPreDetection',
inputs = ["input_1"],
outputs = ["yolo_coord_1", "yolo_obj_1", "yolo_classes_1"],
boxes = boxNum,
coords = coords,
classes = classes,
yolo_version = 'V8',
name = "yolo_pre_node_1")

f_h, f_w = f_h // 2, f_w // 2
crd_align_len = f_h * f_w
obj_align_len = boxNum * f_h * f_w
coord_data_2 = helper.make_tensor_value_info("yolo_coord_2", onnx.TensorProto.FLOAT,
['batch', boxNum * 4, crd_align_len])
obj_prob_2 = helper.make_tensor_value_info("yolo_obj_2", onnx.TensorProto.FLOAT,
['batch', obj_align_len])
classes_prob_2 = helper.make_tensor_value_info("yolo_classes_2", onnx.TensorProto.FLOAT,
['batch', coords, obj_align_len])
yolo_pre_node_2 = helper.make_node('YoloPreDetection',
inputs=["input_2"],
outputs=["yolo_coord_2", "yolo_obj_2", "yolo_classes_2"],
boxes=boxNum,
coords = coords,
classes = classes,
yolo_version='V8',
name="yolo_pre_node_2")

# 创建YOLOv8n检测输出层
img_info = helper.make_tensor_value_info("img_info", onnx.TensorProto.FLOAT, ['batch', img_info_num])
box_out = helper.make_tensor_value_info("box_out", onnx.TensorProto.FLOAT, ['batch', max_boxes_out])
box_out_num = helper.make_tensor_value_info("box_out_num", onnx.TensorProto.INT32, ['batch', box_num])
yolo_detect_node = helper.make_node('YoloV8DetectionOutput',
inputs = [f"yolo_coord_{i}" for i in range(outNUm)] +
[f"yolo_obj_{i}" for i in range(outNUm)] +
[f"yolo_classes_{i}" for i in range(outNUm)] +
['img_info'],
outputs = ['box_out', 'box_out_num'],
boxes = boxNum,
coords = coords,
classes = classes,
pre_nms_topn = pre_nms_topn,
post_nms_topn = post_nms_topn,
relative = relative,
out_box_dim = out_box_dim,
obj_threshold = obj_threshold,
score_threshold = score_threshold,
iou_threshold = iou_threshold,
biases = anchor,
name ='YoloV8DetectionOutput')

# 构建图
graph = helper.make_graph(
nodes = [yolo_pre_node_0, yolo_pre_node_1, yolo_pre_node_2, yolo_detect_node],
name = "yolo",
inputs = [input_0, input_1, input_2, img_info],
outputs = [box_out, box_out_num]
)

# 创建模型
onnx_model = helper.make_model(graph, producer_name="onnx-parser")
onnx_model.opset_import[0].version = 12
onnx.save(onnx_model, "../model/postprocessv8.onnx")
