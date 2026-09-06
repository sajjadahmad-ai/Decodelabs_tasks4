"""
Project 4 - Path 2: Object Detection with MobileNet-SSD
DecodeLabs Industrial Training Kit

This module contains the core, reusable object-detection pipeline:
    1. Load the pre-trained MobileNet-SSD model (Transfer Learning - no training/dataset needed)
    2. Pre-process an input image into a "blob"
    3. Run a forward pass through the network
    4. Filter detections using the 80% confidence threshold (project requirement)
    5. Scale normalized coordinates back to real pixel coordinates
    6. Draw labeled bounding boxes on the image

Both the Jupyter Notebook and the Streamlit app (app.py) import from this file,
so there is a single source of truth for the detection logic.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# STEP 1: Model files (pre-trained - NO dataset / training required)
# ---------------------------------------------------------------------------
# MobileNet-SSD was already trained by researchers on large datasets
# (Transfer Learning). We only need these two files:
#   - MobileNetSSD_deploy.prototxt      -> model architecture/structure
#   - MobileNetSSD_deploy.caffemodel    -> pre-trained weights ("the brain")
#
# Download links (place both files in the "models/" folder):
#   prototxt:   https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt
#   caffemodel: https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel
# (See README.md for exact instructions.)

PROTOTXT_PATH = "models/MobileNetSSD_deploy.prototxt"
MODEL_PATH = "models/MobileNetSSD_deploy.caffemodel"

# The 21 object classes MobileNet-SSD was trained to recognize
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Project 4 requirement: minimum 80% confidence to accept a detection
CONFIDENCE_THRESHOLD = 0.80

# A fixed color per class so boxes are visually consistent
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def load_model(prototxt_path: str = PROTOTXT_PATH, model_path: str = MODEL_PATH):
    """
    STEP 2: Load the pre-trained Caffe model using OpenCV's DNN module.
    This is Transfer Learning in action - we are loading a "brain" that
    has already been trained on millions of images, no dataset needed here.
    """
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net


def preprocess_image(image, size=(300, 300), scale=0.007843, mean=127.5):
    """
    STEP 3: Pre-processing - Blob Construction.
    Converts the raw image into a 4D "blob" that the network expects:
      - Resizes the image to 300x300 (MobileNet-SSD's required input size)
      - Performs mean subtraction (normalizes pixel values)
      - Rearranges data into the (N, C, H, W) format the model needs
    """
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, size),
        scale,
        size,
        mean
    )
    return blob, (h, w)


def run_inference(net, blob):
    """
    STEP 4: Forward Pass - Prediction.
    Feeds the blob into the network and gets raw detections back.
    Each detection includes: class id, confidence score, and normalized
    bounding box coordinates.
    """
    net.setInput(blob)
    detections = net.forward()
    return detections


def filter_and_scale_detections(detections, original_dims, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    STEP 5 + 6: Confidence Filtering + Coordinate Scaling.
    - Drops any detection below the 80% confidence gate (project requirement)
    - Converts normalized (0-1) coordinates into real pixel coordinates
      using the ORIGINAL image's width and height

    Returns a list of dicts: {label, confidence, box: (startX, startY, endX, endY)}
    """
    (h, w) = original_dims
    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # --- The 80% Confidence Gate ---
        if confidence >= confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            results.append({
                "label": CLASSES[idx] if idx < len(CLASSES) else "unknown",
                "confidence": float(confidence),
                "box": (int(startX), int(startY), int(endX), int(endY))
            })

    return results


def draw_detections(image, results):
    """
    STEP 7: Visual Confirmation.
    Draws a labeled, colored bounding box for every accepted detection.
    """
    output = image.copy()

    for det in results:
        (startX, startY, endX, endY) = det["box"]
        idx = CLASSES.index(det["label"]) if det["label"] in CLASSES else 0
        color = COLORS[idx]

        label_text = "{}: {:.2f}%".format(det["label"], det["confidence"] * 100)

        cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(
            output, label_text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return output


def detect_objects(image, net=None, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Convenience wrapper that runs the FULL pipeline end-to-end:
    preprocess -> inference -> filter/scale -> draw

    Returns: (output_image, list_of_detections)
    """
    if net is None:
        net = load_model()

    blob, original_dims = preprocess_image(image)
    detections = run_inference(net, blob)
    results = filter_and_scale_detections(detections, original_dims, confidence_threshold)
    output_image = draw_detections(image, results)

    return output_image, results
