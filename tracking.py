"""
tracking.py
===========

This module encapsulates the core logic required to run object detection and
tracking using the Deep‑SORT algorithm.  It leverages a YOLOv8 model from the
`ultralytics` package for object detection and the `deep_sort_realtime`
implementation for appearance‑based multi‑object tracking.  You can use this
module directly or through the accompanying UI defined in ``main.py``.

The detection component uses pre‑trained YOLOv8 weights to find objects in
individual frames.  These detections are then passed to a Deep SORT tracker,
which maintains consistent IDs by combining motion information from a Kalman
filter with deep appearance features as described in the literature on Deep
SORT【596124166780657†L186-L219】.  Using both motion and appearance allows the
tracker to re‑identify objects after short occlusions or when objects leave
and re‑enter the scene【596124166780657†L186-L219】.

Note:  You must install the required packages listed in requirements.txt before
using this module.  See the documentation in that file for details.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ultralytics is not installed. Please install it via 'pip install ultralytics'"
    ) from e

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "deep-sort-realtime is not installed. Please install it via 'pip install deep-sort-realtime'"
    ) from e


class ObjectTracker:
    """Encapsulates the YOLO detector and Deep SORT tracker."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        classes: Optional[List[int]] = None,
        max_age: int = 30,
        n_init: int = 2,
        max_cosine_distance: float = 0.4,
        nms_max_overlap: float = 1.0,
    ) -> None:
        """
        Initialise the detection and tracking components.

        Parameters
        ----------
        model_path: str, optional
            Path to the YOLOv8 weights (default 'yolov8n.pt').  You can use any
            weight file supported by ``ultralytics``.  The file will be
            automatically downloaded if not present.
        classes: list of int, optional
            If provided, only detections belonging to these class IDs will be
            tracked.  Class IDs correspond to the COCO dataset indices used by
            YOLO models.  Pass ``None`` to track all classes.
        max_age: int, optional
            Maximum number of frames to keep a track alive without associated
            detections.
        n_init: int, optional
            Number of consecutive detections before a track is confirmed.
        max_cosine_distance: float, optional
            Maximum allowed cosine distance for appearance matching.  Lower
            values make the tracker more strict when associating detections to
            existing tracks.
        nms_max_overlap: float, optional
            Maximum allowed overlap for non‑maximum suppression in Deep SORT.
        """

        self.model = YOLO(model_path)
        self.classes = classes

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nms_max_overlap=nms_max_overlap,
        )

    def _prepare_detections(self, frame: np.ndarray) -> List[Tuple[List[float], float, int]]:
        """
        Run the YOLO model on a single frame and convert detections into the
        format expected by the Deep SORT tracker.

        Parameters
        ----------
        frame: np.ndarray
            BGR image array as returned by OpenCV.

        Returns
        -------
        detections: list of tuple
            Each element is a tuple ``([x, y, w, h], confidence, class_id)``.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(rgb_frame, imgsz=640, verbose=False)
        detections: List[Tuple[List[float], float, int]] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes

        xyxy = boxes.xyxy.cpu().numpy()  # shape: (n, 4)
        confidences = boxes.conf.cpu().numpy()  # shape: (n,)
        classes = boxes.cls.cpu().numpy().astype(int)  # shape: (n,)
        for bbox, conf, cls in zip(xyxy, confidences, classes):
            if self.classes is not None and cls not in self.classes:
                continue
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], float(conf), int(cls)))
        return detections

    def update(self, frame: np.ndarray) -> List[Dict[str, object]]:
        """
        Perform detection and update the tracker for a single video frame.

        Parameters
        ----------
        frame: np.ndarray
            BGR image array.

        Returns
        -------
        tracked_objects: list of dict
            Each dictionary contains ``track_id``, ``bbox``, and ``class_id`` keys
            describing the tracked object.
        """
        detections = self._prepare_detections(frame)

        tracks = self.tracker.update_tracks(detections, frame=frame)
        tracked_objects: List[Dict[str, object]] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id

            l, t, r, b = track.to_ltrb()
            class_id = track.det_class if hasattr(track, "det_class") else None
            tracked_objects.append(
                {
                    "track_id": track_id,
                    "bbox": (int(l), int(t), int(r), int(b)),
                    "class_id": class_id,
                }
            )
        return tracked_objects