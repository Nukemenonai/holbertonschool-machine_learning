#!/usr/bin/env python3
"""
Yolo
"""

import tensorflow as tf 
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    uses the Yolo v3 to perform object detection
    """  
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ class constructor """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            classes = f.read().splitlines() 
            self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Returns sigmoid function"""
        return(1/(1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """does something """
        boxes = []
        box_confidence = []
        box_class_probs = []
        i_h, i_w = image_size
        for i, output in enumerate(outputs):
            input_w = self.model.input_shape[1]
            input_h = self.model.input_shape[2]
            g_w, g_h, anchor_boxes, n_classes = output.shape
            t_x = output[..., 0] 
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            
            # corner
            c = np.zeros((g_h, g_w, anchor_boxes))
            # indexes and top-left corner
            idx_y = np.arange(g_h)
            idx_y = idx_y.reshape(g_h, 1, 1)
            idx_x = np.arange(g_w)
            idx_x = idx_x.reshape(1, g_w, 1)
            cx = c + idx_x
            cy = c + idx_y
            
            p_w = self.anchors[i, : , 0]
            p_h = self.anchors[i, : , 1]
            
            bx = self.sigmoid(t_x) + cx #top-left corner width
            by = self.sigmoid(t_y) + cy #top-left corner height
            bw = p_w * np.exp(t_w)
            bh = p_h * np.exp(t_h)

            # normalize bx and by values to the grid
            bx = bx / g_w
            by = by / g_h

            # normalize bw and bh values to the input sizes
            bw = bw / input_w
            bh = bh / input_h

            # get the corner coordinates
            bx1 = bx - bw / 2
            by1 = by - bh / 2
            bx2 = bx + bw / 2
            by2 = by + bh / 2

            # to image size scale
            outputs[i][..., 0] = bx1 * i_w
            outputs[i][..., 1] = by1 * i_h 
            outputs[i][..., 2] = bx2 * i_w
            outputs[i][..., 3] = by2 * i_h

            # filtered bounding boxes
            boxes.append(outputs[i][..., 0:4])
            # objectiveness score between 0 and 1
            box_confidence.append(self.sigmoid(output[..., 4:5]))
            # probability of classes
            box_class_probs.append(self.sigmoid(outputs[i][..., 5:]))
        return (boxes, box_confidence, box_class_probs)
