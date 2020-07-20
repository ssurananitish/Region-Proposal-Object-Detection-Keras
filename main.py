from tensorflow.keras.applications import ResNet50, imagenet_utils
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from utils import non_max_suppression, get_selective_search
import numpy as np
import argparse
import cv2

# Command Line Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True, help="Path for the input image")
ap.add_argument("-m","--method", default="fast", type=str, choices=["fast", "quality"], help="Method for the selective search")
ap.add_argument("-c","--min-conf", type=float, default=0.9, help="Minimum confidence for classification/detection")
ap.add_argument("-f","--filter", type=str, default=None, help="Comma separated list of ImageNet labels to filter on")
args = vars(ap.parse_args())

label_filters = args["filter"]
if label_filters is not None:
    label_filters = label_filters.lower().split(',')

# Loading the model and the image
model = ResNet50(weights="imagenet")
print("Model_loaded")
image = cv2.imread(args["image"])
(H,W) = image.shape[:2]

# Applying selective search on the image
rects = get_selective_search(image, method=args["method"])
proposals, boxes = [], []

for (x,y,w,h) in rects:
    # Any region having having height or width less than 10% are considered as small regions and ignored
    if w/float(W) < 0.1  or x/float(H) < 0.1:
        continue
    
    roi = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi,(224,224))
    proposals.append(preprocess_input(img_to_array(roi)))
    boxes.append((x,y,w,h))

proposals = np.array(proposals)
preds = model.predict(proposals)
print("{} region of interests".format(len(preds)))
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}

for (i,p) in enumerate(preds):
    (imagenet_ID, label, prob) = p[0]

    if label_filters is not None and label not in label_filters:
        continue

    if prob >= args["min_conf"]:
        (x,y,w,h) = boxes [i]
        box = (x,y,x+w,y+h)
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L


for label in labels.keys():
    print("Results for '{}'".format(label))
    clone = image.copy()
    for (box, prob) in labels[label]:
        (x1, y1, x2, y2) = box
        cv2.rectangle(clone, (x1, y1), (x2, y2),(0, 255, 0), 2)

    cv2.imshow("Before_NMS", clone)
    clone = image.copy()
    boxes, proba = np.array([p[0] for p in labels[label]]), np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(clone, (x1, y1), (x2, y2),(0, 255, 0), 2)
        y = y1-10 if y1-10 > 10 else y1 + 10
        cv2.putText(clone, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    cv2.imshow("After_NMS", clone)
    cv2.imwrite("./Output_images/Output.jpg", clone)
    cv2.waitKey(0)

