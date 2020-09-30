#!/usr/bin/env python3
"""
Yolo
"""

import tensorflow.keras as K


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
