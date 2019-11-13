# import required packages
import cv2
import argparse
import numpy as np
import time

class plasticClassifier:

    config = 'resources/yolov3-obj.cfg'
    weights = 'resources/yolov3-obj_final.weights'
    object_classes = 'resources/obj.names'
    file_object = open("BoundingBoxes.txt", "w")
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
        
    def set_path(self, path):
        self.path = path

    def set_config(self, config_path):
        self.config = config_path

    def set_weights(self, weights_path):
        self.weights = weights_path

    def set_classes(self, classes_path):
        self.object_classes = classes_path

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # function to draw bounding box on the detected object with class name
    def output_bounding_box(self, img, class_id, confidence, x, y, w, h):
        label = str(self.classes[class_id])
        confidence_percentage = confidence * 100
        label += ' {0:.1f}%'.format(confidence_percentage)
        color = self.COLORS[class_id]
        round_x = round(x)
        round_y = round(y)
        x_plus_w = round(x+w)
        y_plus_h = round(y+h)
        center_x = round(x + w/2)
        center_y = round(y + h/2)
        cv2.rectangle(img, (round_x,round_y), (x_plus_w,y_plus_h), color, 3)
        cv2.putText(img, label, (round_x-10,round_y-10), cv2.FONT_HERSHEY_TRIPLEX, 1.2, color, 2)
        self.file_object.write(label + "\n")
        self.file_object.write('Center X: %s\nCenter Y: %s\n' % (center_x, center_y))
        self.file_object.write('Width: %s\nHeight: %s\n\n' % (round(w), round(h)))

    def label_image(self):

        # Keeps track of execution time.
        start_time = time.time()

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
                if confidence > self.conf_threshold:
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


        # go through the remaining detections
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            
            self.output_bounding_box(image, class_ids[i], confidences[i], x, y, w, h)   
        # save output image to disk
        cv2.imwrite("object-detection.jpg", image)

        print('Executed in %s seconds.' % (time.time() - start_time))

if __name__ == "__main__":
    conf = 0.5
    nms = 0.3
    ap = argparse.ArgumentParser(
        description='''Detects objects in an image using a pre-trained Yolov3 model'''
    )
    ap.add_argument('-i', '--image', required=False, help = 'Path to image file.') #path to input image
    ap.add_argument('-cf', '--confidence', required=False, help = 'Custom Confidence level. Default: 0.5')
    ap.add_argument('-n', '--nms', required=False, help = 'Custom Non-Maximum Supression level. Default: 0.3')
    ap.add_argument('-c', '--config', required=False, help = 'Path to cfg file. Default: resources/yolov3-obj.cfg') # path to cfg
    ap.add_argument('-w', '--weights', required=False, help = 'Path to weights file. Default: resources/yolov3-obj_final.weights') #path to .weights
    ap.add_argument('-cl', '--classes', required=False, help = 'Path to class names file. Default: resources/obj.names') #path to .txt with class names
    args = ap.parse_args()

    if args.confidence is not None:
        conf = args.confidence

    if args.nms is not None:
        nms = args.nms

    classifier = plasticClassifier(conf, nms, args.image)

    if args.config is not None:
        classifier.set_config(args.config)
    
    if args.weights is not None:
        classifier.set_weights(args.weights)
    
    if args.classes is not None:
        classifier.set_classes(args.classes)
    
    classifier.label_image()