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

            c = np.zeros((g_h, g_w, anchor_boxes))
            idx_y = np.arange(g_h)
            idx_y = idx_y.reshape(g_h, 1, 1)
            idx_x = np.arange(g_w)
            idx_x = idx_x.reshape(1, g_w, 1)
            cx = c + idx_x
            cy = c + idx_y

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            bx = self.sigmoid(t_x) + cx
            by = self.sigmoid(t_y) + cy
            bw = p_w * np.exp(t_w)
            bh = p_h * np.exp(t_h)

            bx = bx / g_w
            by = by / g_h

            bw = bw / input_w
            bh = bh / input_h

            bx1 = bx - bw / 2
            by1 = by - bh / 2
            bx2 = bx + bw / 2
            by2 = by + bh / 2

            outputs[i][..., 0] = bx1 * i_w
            outputs[i][..., 1] = by1 * i_h
            outputs[i][..., 2] = bx2 * i_w
            outputs[i][..., 3] = by2 * i_h

            boxes.append(outputs[i][..., 0:4])
            box_confidence.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(outputs[i][..., 5:]))
        return (boxes, box_confidence, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ filter_boxes """
        res = []
        for i in range(len(boxes)):
            res.append(box_class_probs[i] * box_confidences[i])

        idx_class = [np.argmax(elem, -1) for elem in res]
        idx_class = [elem.reshape(-1) for elem in idx_class]
        idx_class = np.concatenate(idx_class)

        score_class = [np.max(elem, axis=-1) for elem in res]
        score_class = [elem.reshape(-1) for elem in score_class]
        score_class = np.concatenate(score_class)

        filter_box = [elem.reshape(-1, 4) for elem in boxes]
        filter_box = np.concatenate(filter_box)

        filter = np.where(score_class > self.class_t)

        box_classes = idx_class[filter]
        box_scores = score_class[filter]
        filter_boxes = filter_box[filter]
        return (filter_boxes, box_classes, box_scores)
