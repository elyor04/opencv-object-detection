from cv2 import (
    VideoCapture,
    rectangle,
    putText,
    imshow,
    getTextSize,
    waitKey,
    destroyAllWindows,
    FONT_HERSHEY_COMPLEX_SMALL,
)
from cv2.dnn import DetectionModel
from os import getcwd
from os.path import join, abspath
from time import time
from random import randint
from numpy import ndarray
from typing import Iterable


ColorsType = dict[str, tuple[int, int, int]]
colors: ColorsType = dict()


def visualize_boxes_and_labels(
    image: ndarray,
    boxes: Iterable[Iterable[int]],
    class_ids: Iterable[int],
    scores: Iterable[float],
    class_names: dict[int, str],
) -> ndarray:
    height, width = image.shape[:2]
    for (xmin, ymin, wd, hg), cls_id, score in zip(boxes, class_ids, scores):
        perc = int(score * 100)
        if perc < 60:
            continue
        xmax, ymax = xmin + wd, ymin + hg
        name = class_names[cls_id].capitalize()

        if name in colors:
            color = colors[name]
        else:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            while (True not in [(pxl > 210) for pxl in color]) or (
                color in colors.values()
            ):
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
            colors[name] = color

        # name = f"{name} {perc}%"
        gts = getTextSize(name, FONT_HERSHEY_COMPLEX_SMALL, 2.0, 2)
        gtx = gts[0][0] + xmin
        gty = gts[0][1] + ymin

        rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
        image[ymin : min(gty + 4, height), xmin : min(gtx + 3, width)] = color
        putText(image, name, (xmin, gty), FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 0, 0), 2)
    return image


cap = VideoCapture(0)

data_dir = join(getcwd(), abspath("data/models/ssd_mobilenet_v3_large_coco_2020_01_14"))
classFile = join(data_dir, "coco.names")
configPath = join(data_dir, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
weightsPath = join(data_dir, "frozen_inference_graph.pb")

with open(classFile, "rt") as f:
    classNames = f.read().strip("\n").splitlines()
    classNames = {(id + 1): name for (id, name) in enumerate(classNames)}

net = DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

prevTime = 0

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.45)

    if len(classIds) != 0:
        visualize_boxes_and_labels(
            img, bbox, classIds.flatten(), confs.flatten(), classNames
        )

    currTime = time()
    fps = f"FPS: {round(1 / (currTime - prevTime), 1)}"
    prevTime = currTime

    putText(img, fps, (5, 35), FONT_HERSHEY_COMPLEX_SMALL, 2.0, (255, 0, 0), 2)
    imshow("Object Detection", img)

    if waitKey(2) == 27:
        break

cap.release()
destroyAllWindows()
