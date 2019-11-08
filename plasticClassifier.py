# import required packages
import cv2
import argparse
import numpy as np

class plasticClassifier:

    config = 'resources/yolov3-obj.cfg'
    weights = 'resources/yolov3-obj_last.weights'
    object_classes = 'resources/obj.names'

    conf_threshold = 0.5
    nms_threshold = 0.3
    path = ''
    classes = None
    COLORS = None

    def __init__ (self, conf, nms, path):
        self.conf_threshold = conf
        self.nms_threshold = nms
        self.path = path

        with open(self.object_classes, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 3)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)

    def label_image(self):
        # read input image
        image = cv2.imread(self.path)

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        # read pre-trained model and config file
        net = cv2.dnn.readNet(self.weights, self.config)

        # create input blob 
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)

        # function to get the output layer names 
        # in the architecture
        outs = net.forward(self.get_output_layers(net))

        # run inference through the network
        # and gather predictions from output layers
        # initialization
        class_ids = []
        confidences = []
        boxes = []

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            
            self.draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))   
        # save output image to disk
        cv2.imwrite("object-detection.jpg", image)