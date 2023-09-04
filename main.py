import cv2 as cv
import numpy as np
import os.path as path
from json import load
from time import time
from random import randint
from typing import Iterable


colors: dict[str, tuple[int, int, int]] = dict()


def visualize_boxes_and_labels(
    image: np.ndarray,
    boxes: Iterable[Iterable[int]],
    class_ids: Iterable[int],
    scores: Iterable[float],
    class_names: dict[int, str],
) -> np.ndarray:
    height, width = image.shape[:2]

    for (xmin, ymin, wd, hg), cls_id, score in zip(boxes, class_ids, scores):
        perc = int(score * 100)
        if perc < 60:
            continue
        xmax, ymax = xmin + wd, ymin + hg
        name = class_names[str(cls_id)].capitalize()

        if name in colors:
            color = colors[name]
        else:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            while (True not in [(pxl > 210) for pxl in color]) or (
                color in colors.values()
            ):
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
            colors[name] = color

        name = f"{name} {perc}%"
        font = cv.FONT_HERSHEY_COMPLEX_SMALL

        gts = cv.getTextSize(name, font, 2.0, 2)
        gtx = gts[0][0] + xmin
        gty = gts[0][1] + ymin

        cv.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
        cv.rectangle(
            image, (xmin, ymin), (min(gtx + 3, width), min(gty + 4, height)), color, -1
        )
        cv.putText(image, name, (xmin, gty), font, 2.0, (0, 0, 0), 2)

    return image


class DetectionModel(cv.dnn.DetectionModel):
    def __init__(self, dataDir: str = "data") -> None:
        self.DATA_DIR = dataDir

        self.WEIGHTS_PATH = path.join(self.DATA_DIR, "frozen_inference_graph.pb")
        self.CONFIG_PATH = path.join(
            self.DATA_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        )
        self.LABELS_PATH = path.join(self.DATA_DIR, "coco_names.json")

        self.names = None

    def loadModel(self) -> None:
        super().__init__(self.WEIGHTS_PATH, self.CONFIG_PATH)

    def loadNames(self) -> None:
        with open(dm.LABELS_PATH, "rt") as _f:
            self.names = load(_f)

    def prepareAll(self) -> None:
        self.loadModel()
        self.loadNames()
        self.setInputSize(320, 320)
        self.setInputScale(1.0 / 127.5)
        self.setInputMean((127.5, 127.5, 127.5))
        self.setInputSwapRB(True)


dm = DetectionModel()
dm.prepareAll()

cap = cv.VideoCapture(0)
prevTime = 0

while True:
    success, img = cap.read()
    classIds, confs, bbox = dm.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        visualize_boxes_and_labels(
            img, bbox, classIds.flatten(), confs.flatten(), dm.names
        )

    currTime = time()
    fps = f"FPS: {round(1 / (currTime - prevTime), 1)}"
    prevTime = currTime

    cv.putText(img, fps, (5, 35), cv.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (255, 0, 0), 2)
    cv.imshow("Object Detection", img)

    if cv.waitKey(2) == 27:  # esc
        break

cap.release()
cv.destroyAllWindows()
