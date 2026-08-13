import json
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import sys

def test_model():
    with open("model/crnn.json", "r") as f:
        meta = json.load(f)

    alphabet = meta["alphabet"]
    img_w, img_h = meta["img_w"], meta["img_h"]

    session = ort.InferenceSession("model/crnn.onnx")
    print(f"Model loaded. Alphabet: {alphabet}")

test_model()
