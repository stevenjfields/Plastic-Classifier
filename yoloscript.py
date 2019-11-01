import cv2
import argparse
import numpy as np

# run this script by using the command: python yoloscript.py --image "yourimage".jpg --config "yourconfig".cfg --weights "yourweights".weights --classes "yourclasses".txt
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help = '') #path to input image
ap.add_argument('-c', '--config', required=True, help = '') # path to cfg
ap.add_argument('-w', '--weights', required=True, help = '') #path to .weights
ap.add_argument('-cl', '--classes', required=True, help = '') #path to .txt with class names
args = ap.parse_args()

# read input image
image = cv2.imread(args.image)

imageWidth = image.shape[1]
imageHeight = image.shape[0]
imageScale = 0.00450

# read class names from text file
classes = None
with open(args.classes, 'r') as f: 
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
boxColor = np.random.uniform(0, 255, size=(len(classes), 3))

# read .weights and cfg
net = cv2.dnn.readNet(args.weights, args.config)

# create input blob and set for the network
blob = cv2.dnn.blobFromImage(image, imageScale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)

#get the output layer names 
def get_output_layers(net):

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# draw bounding box on detected object
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    color = boxColor[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



# for each detetion from each output layer 
#get the confidence (confidence < 0.5), class id, bounding box params
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * imageWidth)
            h = int(detection[3] * imageHeight)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

cv2.imshow("Detected Objects: ", image)
