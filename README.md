# Project 4 – Object Detection (MobileNet-SSD)
DecodeLabs Industrial Training Kit — Batch 2026
Object Detection

---

## 📌 Project Overview

This project implements a basic **Object Detection** pipeline using a
**pre-trained** MobileNet-SSD model (Transfer Learning — no dataset or
model training required). The script:

1. Loads the pre-trained model
2. Pre-processes a sample image into a "blob"
3. Runs a forward pass (prediction)
4. Filters detections using an **80% minimum confidence threshold**
5. Scales coordinates back to real pixel values
6. Draws labeled bounding boxes on the image and displays the result

---

## 📂 Folder Structure

```
Object Detection (MobileNet-SSD)/
├── app.py                             # Streamlit web app (interactive demo)
├── detector.py                        # Core detection pipeline (shared logic)
├── Project4_Object_Detection.ipynb    # Jupyter Notebook (step-by-step)
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── MobileNetSSD_deploy.prototxt       # Pre-trained model architecture
├── MobileNetSSD_deploy.caffemodel     # Pre-trained model weights
└── <your_test_image>.jpg              # Sample image for testing
```

> **Note:** All files are kept directly inside the main project folder
> (no separate `models/` subfolder) — paths in `detector.py` /
> `Project4_Object_Detection.ipynb` are set accordingly.

---

## ⚙️ Requirements / Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV (`cv2`) | Image processing + running the deep learning model |
| `cv2.dnn` module | Loading and running the pre-trained Caffe model |
| MobileNet-SSD | Pre-trained object detection model (20 object classes) |
| NumPy | Array/matrix operations |
| Streamlit | Interactive web app for uploading images and viewing results |
| Jupyter Notebook | Step-by-step demonstration of the pipeline |

Install everything with:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Files (Pre-Trained — No Dataset Needed)

This project uses **Transfer Learning**: MobileNet-SSD was already trained
by researchers on large image datasets. We only use its two files:

- `MobileNetSSD_deploy.prototxt` — model architecture
- `MobileNetSSD_deploy.caffemodel` — pre-trained weights

(Already included in this project folder.)

**Classes it can detect (20 objects):**
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
train, tvmonitor

---

## ▶️ How to Run

### Option A — Jupyter Notebook (step-by-step walkthrough)

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Project4_Object_Detection.ipynb`
3. Update `IMAGE_PATH` in the notebook to point to your test image
   (e.g. `IMAGE_PATH = "car.jpg"`)
4. Run all cells in order

### Option B — Streamlit App (interactive browser demo)

1. Open **Command Prompt (CMD)**
2. Navigate to the project folder:
   ```cmd
   cd "C:\Users\AL REHMAN LAPTOPS\Documents\Object Detection (MobileNet-SSD)"
   ```
3. Run the app:
   ```cmd
   streamlit run app.py
   ```
4. The app opens automatically in your browser at `http://localhost:8501`
5. Upload any image to see detected objects with bounding boxes and
   confidence scores

---

## ✅ How It Works (Summary)

| Step | What Happens |
|---|---|
| 1. Load Model | `cv2.dnn.readNetFromCaffe()` loads the pre-trained network |
| 2. Load Image | `cv2.imread()` reads the sample test image |
| 3. Pre-Process | `cv2.dnn.blobFromImage()` resizes to 300x300 and normalizes |
| 4. Predict | `net.forward()` returns raw detections |
| 5. Filter | Only detections with confidence ≥ **80%** are kept |
| 6. Scale | Normalized coordinates converted to real pixel coordinates |
| 7. Draw | `cv2.rectangle()` + `cv2.putText()` draw labeled boxes |
| 8. Output | Result displayed / saved as an image |

---

## 🎯 Project 4 Validation Checklist

- [x] **Library Integration** — uses `cv2.dnn` with MobileNet-SSD
- [x] **Pre-Processing Integrity** — blob construction (resize + mean subtraction)
- [x] **Accuracy Benchmarking** — 80% minimum confidence filter applied
- [x] **Visual Confirmation** — labeled bounding boxes drawn on final output

---

## 🧑‍💻 Author
Submitted as part of DecodeLabs Artificial Intelligence — Industrial Training Kit,
Batch 2026.
