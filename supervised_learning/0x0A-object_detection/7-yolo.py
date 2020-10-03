#!/usr/bin/env python3
"""
Yolo
"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2


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

            p_filters = filters[pick]
            p_scores = scores[pick]
            p_classes = classes[pick]

            box_pred.append(p_filters)
            box_classes_pred.append(p_classes)
            box_scores_pred.append(p_scores)

        filtered_boxes = np.concatenate(box_pred, axis=0)
        box_classes = np.concatenate(box_classes_pred, axis=0)
        box_scores = np.concatenate(box_scores_pred, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def _iou(self, filtered_boxes, thresh, scores):
        """ does iou """
        x1 = filtered_boxes[:, 0]
        y1 = filtered_boxes[:, 1]
        x2 = filtered_boxes[:, 2]
        y2 = filtered_boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = scores.argsort()[::-1]

        pick = []
        while idxs.size > 0:
            i = idxs[0]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            overlap = inter / (area[i] + area[idxs[1:]] - inter)
            ind = np.where(overlap <= thresh)[0]
            idxs = idxs[ind + 1]

        return pick

    @staticmethod
    def load_images(folder_path):
        """ folder_path: a string representing
        the path to the folder holding all the images to load
        Returns a tuple of (images, image_paths):
        images: a list of images as numpy.ndarrays
        image_paths: a list of paths to the individual images in images
        """
        imgs = []
        img_path = glob.glob(folder_path + '/*.jpg', recursive=False)
        for img in img_path:
            image = cv2.imread(img)
            imgs.append(image)
        return (imgs, img_path)

    def preprocess_images(self, images):
        """preprocess images"""
        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value
        dims = (input_w, input_h)
        process_img = []
        original_dims = []
        for img in images:
            original = np.array([img.shape[0], img.shape[1]])
            original_dims.append(original)
            # inter-cubic interpolation resize
            img = cv2.resize(img, dims, interpolation=cv2.INTER_CUBIC)
            # rescale to pixels [0 - 1]
            img = img / 255
            process_img.append(img)

        # adding one shape in order to do the concatenate still have the 4 dims
        p_images = [el.reshape(1, input_h, input_w, 3) for el in process_img]
        p_images = np.concatenate(p_images)

        # adding one shape in order to do the concatenate still have the 2 dims
        image_shapes = [el.reshape(-1, 2) for el in original_dims]
        image_shapes = np.concatenate(image_shapes)

        return (p_images, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """show detection boxes"""

        color = (0, 0, 255)
        for i in range(len(boxes)):
            x1 = int(boxes[i][0])
            y1 = int(boxes[i][1])
            x2 = int(boxes[i][2])
            y2 = int(boxes[i][3])

            cls = self.class_names[box_classes[i]]
            score = float(box_scores[i])
            label = cls + ' {:0.2f}'.format(score)

            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText(image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if not os.path.exists('detections'):
                os.mkdir('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()


   def predict(self, folder_path):
        """Predicts bounding boxes for all images"""
        predictions = []
        images, images_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        outputs = self.model.predict(pimages)
        o1 = outputs[0][:]
        o2 = outputs[1][:]
        o3 = outputs[2][:]
        for i in range(len(images)):
            boxes, box_confidences, box_class_probs = self.process_outputs(
                [o1[i],
                 o2[i],
                 o3[i]],
                image_shapes[i])
            boxes, box_cls, box_sc = self.filter_boxes(boxes,
                                                               box_confidences,
                                                               box_class_probs)
            boxes, box_classes, box_scores = self.non_max_suppression(boxes,
                                                                      box_cls,
                                                                      box_sc)
            self.show_boxes(images[i], boxes, box_classes, box_scores,
                            images_paths[i].split('/')[-1])
            predictions.append((boxes, box_classes, box_scores))
        return(predictions, images_paths)
