import cv2
from time import time
import numpy as np
import onnxruntime
from matplotlib import pyplot as plt


class YOLOX_ONNX:

    def __init__(self, model_path):
        providers = ['CPUExecutionProvider']
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.image_size = self.model.get_inputs()[0].shape[-2:]
        # print(self.model.get_outputs()[0].name)
        # print(self.image_size)
        self.labels_map=['pedestrian']


        

    def __preprocess_image(self, img, swap=(2, 0, 1)):

        padded_img = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 114
        r = min(self.image_size[0] / img.shape[0], self.image_size[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r


    @staticmethod
    def __new_nms(boxes, scores, iou_thresh):
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
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep


    def __parse_output_data(self, outputs):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32]
        hsizes = [self.image_size[0] // stride for stride in strides]
        wsizes = [self.image_size[1] // stride for stride in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs[0]

    def __decode_prediction(self, prediction, img_size, resize_ratio, score_thresh, iou_thresh):

        boxes = prediction[:, :4]
        classes = prediction[:, 4:5] * prediction[:, 5:]
        scores = np.amax(classes, axis=1)
        classes = np.argmax(classes, axis=1)


        valid_score_mask = scores > score_thresh
        if valid_score_mask.sum() == 0:
            return np.array([]), np.array([]), np.array([])
        valid_scores = scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_classes = classes[valid_score_mask]


        valid_boxes_xyxy = np.ones_like(valid_boxes)
        valid_boxes_xyxy[:, 0] = valid_boxes[:, 0] - valid_boxes[:, 2]/2.
        valid_boxes_xyxy[:, 1] = valid_boxes[:, 1] - valid_boxes[:, 3]/2.
        valid_boxes_xyxy[:, 2] = valid_boxes[:, 0] + valid_boxes[:, 2]/2.
        valid_boxes_xyxy[:, 3] = valid_boxes[:, 1] + valid_boxes[:, 3]/2.
        valid_boxes_xyxy /= resize_ratio

        indices = self.__new_nms(valid_boxes_xyxy, valid_scores, iou_thresh)
        valid_boxes_xyxy = valid_boxes_xyxy[indices, :]
        valid_scores = valid_scores[indices]
        valid_classes = valid_classes[indices].astype('int')
        
        #valid_boxes_xyxy, valid_scores, valid_classes = self.__remove_duplicates(valid_boxes_xyxy, valid_scores, valid_classes)


        return valid_boxes_xyxy, valid_scores, valid_classes

    def draw_boxes(self,img, boxes, scores=None, classes=None, labels=None):


        for i in range(boxes.shape[0]):
            cv2.rectangle(img,
                        (int(boxes[i,0]), int(boxes[i,1])), 
                        (int(boxes[i,2]), int(boxes[i,3])),
                        (0, 128, 0),
                        int(0.005*img.shape[1]))

            ### not drawing classes since num_classes is 1(pedestrian) and text not greatly visible in gradio UI
            # text_label = ''
            # if labels is not None:
            #     if classes is not None:
            #         text_label = labels[classes[i]]
            #     if scores is not None:
            #         text_label+= ' ' + str("%.2f" % round(scores[i],2))
            # elif scores is not None:
            #     text_label = str("%.2f" % round(scores[i],2))

            #w, h = cv2.getTextSize(text_label, 0, fontScale=0.5, thickness=1)[0]
            # cv2.putText(img,
            #             text_label,
            #             (int(boxes[i,0]) if int(boxes[i,0])+w<img.shape[1] else img.shape[1]-w, int(boxes[i,1])-2 if (int(boxes[i,1])-h>=3) else int(boxes[i,1])+h+2),
            #             0,
            #             0.5,
            #             (0,0,255),
            #             thickness= int(0.005*img.shape[1]),
            #             lineType=cv2.LINE_AA)
        return img

    def predict(self, image, score_thresh=0.5, iou_thresh=0.4):
        h,w = image.shape[:2]
        origin_img=np.copy(image)
        model_input = np.copy(image)
        model_input, resize_ratio = self.__preprocess_image(model_input)
        #print(model_input.shape)
        #print('input mean:', np.mean(model_input))
        start_time=time()
        prediction = self.model.run(None, {self.model.get_inputs()[0].name: model_input[None, :, :, :]})
        #print(self.model.get_inputs()[0].name)
        #print('output shape:',prediction[0].shape)
        prediction = self.__parse_output_data(prediction[0])
        d_boxes, d_scores, d_classes=self.__decode_prediction(prediction, (h,w), resize_ratio, score_thresh, iou_thresh)
        self.output_img = self.draw_boxes(origin_img, d_boxes,None, d_classes, self.labels_map)
        print('elapsed time:',time()-start_time)
                
        return d_boxes, d_scores, d_classes

if __name__=="__main__":
    path='test-images/test1.jpg'
    yolox_nano_onnx=YOLOX_ONNX('models/pedestrian-detection-best95.onnx')
    yolox_nano_onnx.predict(cv2.imread(path))
    plt.title('Predicted')
    plt.imshow(cv2.cvtColor(yolox_nano_onnx.output_img,cv2.COLOR_BGR2RGB))
    plt.show()
