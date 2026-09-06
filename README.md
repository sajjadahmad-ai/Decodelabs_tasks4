# Project 4 – Path 2: Object Detection (MobileNet-SSD)
DecodeLabs Industrial Training Kit — Batch 2026

## What's in this folder

```
project4/
├── detector.py                        # Core detection pipeline (shared logic)
├── app.py                             # Streamlit web app (interactive demo)
├── Project4_Object_Detection.ipynb    # Jupyter Notebook (step-by-step walkthrough)
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── models/                            # <-- YOU create this, put model files here
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
└── sample_images/                     # <-- YOU create this, put test images here
    └── test1.jpg
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Download the pre-trained model files (NO dataset/training needed)

This project uses **Transfer Learning** — the model is already trained.
Create a `models/` folder and download these 2 files into it:

- **MobileNetSSD_deploy.prototxt** (model architecture)
  https://github.com/chuanqi305/MobileNet-SSD/blob/master/deploy.prototxt

- **MobileNetSSD_deploy.caffemodel** (pre-trained weights, ~23MB)
  https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel

  (Rename `mobilenet_iter_73000.caffemodel` to `MobileNetSSD_deploy.caffemodel`
  after downloading, or update the path in `detector.py`.)

> Note: GitHub sometimes needs "raw" download links — if the direct link doesn't
> download the file, look for a "Download raw file" / "raw.githubusercontent.com"
> option on the page.

## 3. Add a sample test image

Put any image containing common objects (person, car, dog, chair, bottle, etc.)
into a `sample_images/` folder, e.g. `sample_images/test1.jpg`.

## 4. Run it

**Option A — Jupyter Notebook** (step-by-step, good for understanding/submission):
```bash
jupyter notebook Project4_Object_Detection.ipynb
```

**Option B — Streamlit App** (interactive, upload any image via browser):
```bash
streamlit run app.py
```

## How it works (summary)

1. Load pre-trained MobileNet-SSD model (`cv2.dnn.readNetFromCaffe`)
2. Pre-process input image into a blob (`cv2.dnn.blobFromImage`)
3. Run forward pass → get raw detections
4. Keep only detections with **confidence >= 80%** (project requirement)
5. Scale normalized coordinates to real pixel coordinates
6. Draw labeled bounding boxes and display/save the result

## Project 4 Validation Checklist

- [x] Library Integration — uses `cv2.dnn` with MobileNet-SSD
- [x] Pre-Processing Integrity — blob construction (resize, mean subtraction)
- [x] Accuracy Benchmarking — 80% minimum confidence filter applied
- [x] Visual Confirmation — labeled bounding boxes drawn on output image
