import numpy as np 
import cv2

def get_selective_search(image, method="fast"):
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(image)

    if method == "fast":
        selective_search.switchToSelectiveSearchFast()
    else:
        selective_search.switchToSelectiveSearchQuality()

    rects = selective_search.process()
    return rects

def non_max_suppression(boxes, probs=None, overlapping_threshold=0.3):
    # Function for implementing the non max suppression and selecting the bounding box with maximum confidence
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    selected_indexes = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if probs is not None: 
        idxs = probs
    else: 
        idxs = y2

    idxs = np.argsort(idxs)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        selected_indexes.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[idxs[:last]]), np.maximum(y1[i], y1[idxs[:last]])
        xx2, yy2 = np.minimum(x2[i], x2[idxs[:last]]), np.minimum(y2[i], y2[idxs[:last]])
        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapping_threshold)[0])))
    
    return boxes[selected_indexes].astype("int")