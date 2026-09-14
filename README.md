# Deep SORT Object Tracking Project

This project implements a complete pipeline for multi‑object tracking based on the **Deep SORT** algorithm.  It combines a modern YOLOv8 detector with the Deep SORT tracker to robustly follow objects through a video sequence while maintaining a consistent identity for each object over time.  A minimal graphical user interface (GUI) built with Tkinter allows you to choose a video file or live webcam feed, start and stop tracking, and view the results in real time.

## Background

**Deep SORT** (“Deep Simple Online and Realtime Tracking”) extends the original SORT algorithm by incorporating a deep appearance descriptor.  Deep SORT first uses an object detector—typically a convolutional neural network such as YOLO—to localize objects in each frame【596124166780657†L186-L219】.  For each detection, it extracts a high‑dimensional appearance embedding using another CNN, then uses a Kalman filter to predict object motion and the Hungarian algorithm to associate current detections to existing tracks based on a cost matrix that combines motion and appearance cues【596124166780657†L186-L219】.  This fusion of motion and appearance information makes Deep SORT robust to occlusions and identity switches.

## Features

- **Detection**: Utilises Ultralytics’ YOLOv8 models, which you can swap by changing the model weight file.
- **Tracking**: Employs the `deep‑sort‑realtime` implementation of Deep SORT with configurable parameters (e.g. maximum track age and appearance distance threshold).
- **GUI**: Simple Tkinter interface to select a video file or default to the webcam, start/stop tracking, and exit the application.
- **Threading**: Tracking runs in a separate thread to keep the GUI responsive.
- **Extensible**: The modular design lets you integrate different detectors or modify the tracker settings easily.

## Installation

1. **Clone or download** this repository.
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   The `ultralytics` package will automatically download the default YOLOv8 weights (`yolov8n.pt`) the first time you run the application.  You can provide your own weights file via the `ObjectTracker` constructor in `tracking.py`.

3. **Optional**: If Tkinter is not available on your system, install it via your OS package manager.  For Debian/Ubuntu this can be done with:

   ```bash
   sudo apt install python3‑tk
   ```

## Usage

Run the application with Python 3:

```bash
python3 main.py
```

Upon launching, click **“Select Video”** to choose a video file.  If you cancel the file dialog, the application will default to using your primary webcam.  Press **“Start”** to begin tracking.  Detected objects will be displayed with coloured bounding boxes and ID labels.  Use **“Stop”** to pause tracking or **“Quit”** to exit the program.

## Project Structure

| File            | Description                                     |
|-----------------|-------------------------------------------------|
| `main.py`       | Tkinter GUI that drives the tracking pipeline. |
| `tracking.py`   | Encapsulates detection and Deep SORT tracking. |
| `requirements.txt` | List of Python dependencies.                |
| `README.md`     | This overview and usage guide.                 |

## Customising

- **Model selection**: In `tracking.py`, change the `model_path` argument when instantiating `ObjectTracker` to use a different YOLO variant (e.g. `yolov8s.pt`) or your own trained weights.
- **Class filtering**: Pass a list of class IDs to the `classes` parameter of `ObjectTracker` to restrict tracking to specific object categories.  For example, to track only persons (COCO class 0) and cars (class 2), use `classes=[0, 2]`.
- **Tracker parameters**: Adjust `max_age`, `n_init`, `max_cosine_distance`, and `nms_max_overlap` in `ObjectTracker` to fine‑tune track management.

## Further Reading

If you’re curious about how Deep SORT works under the hood, the Ikomia blog provides a thorough explanation of its components: detection and feature extraction, Kalman filtering for state prediction, and data association that combines motion and appearance cues【596124166780657†L186-L219】.  These ideas build upon the original SORT algorithm and enable reliable tracking even when objects are temporarily occluded【596124166780657†L186-L219】.