from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import cv2


def extract_faces(img) -> List[Tuple[int, int, int, int]]:
    # simple Haar cascade fallback (no heavy deps)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(32, 32))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
