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
        """ """
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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """non max suppression"""

        box_pred = []
        box_classes_pred = []
        box_scores_pred = []
        u_classes = np.unique(box_classes)
        for cls in u_classes:
            idx = np.where(box_classes == cls)

            filters = filtered_boxes[idx]
            scores = box_scores[idx]
            classes = box_classes[idx]

            pick = self._iou(filters, self.nms_t, scores)

            filters = filters[pick]
            scores = scores[pick]
            classes = classes[pick]

            box_pred.append(filters)
            box_classes_pred.append(classes)
            box_scores_pred.append(scores)
        filtered_boxes = np.concatenate(box_pred, axis=0)
        box_classes = np.concatenate(box_classes_pred, axis=0)
        box_scores = np.concatenate(box_scores_pred, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    
    def _iou(self, filtered_boxes, thresh, scores):
 
        # grab the coordinates of the bounding boxes
        x1 = filtered_boxes[:,0]
        y1 = filtered_boxes[:,1]
        x2 = filtered_boxes[:,2]
        y2 = filtered_boxes[:,3]
 
	    # compute the area of the bounding boxes and sort the bounding
    	# boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = scores.argsort()[::-1]

        # initialize the list of picked indexes	
        pick = []
        # keep looping while some indexes still remain in the indexes
	    # list
        while idxs.size > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            i = idxs[0]
            pick.append(i)
 
		    # find the largest (x, y) coordinates for the start of
		    # the bounding box and the smallest (x, y) coordinates
		    # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
 
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
 
            # compute the ratio of overlap
            inter = (w * h)
            overlap =  inter / (area[i] + area[idxs[1:]])
 
            # delete all indexes from the index list that have
            ind = np.where(overlap > self.nms_t)[0]
            idxs = idxs[ind + 1]

 
        # return only the bounding boxes that were picked using the
        # integer data type
        return pick
