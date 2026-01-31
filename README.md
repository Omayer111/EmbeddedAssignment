# YOLOv5 Object Detection - Simple & IPC Implementations

## Overview

This project provides **two approaches** for object detection using YOLOv5:

1. **Simple Detection** (`detect.py`) - All-in-one, perfect for Windows
2. **IPC Detection** - Multi-process with shared memory (C + Python, for embedded systems)

## Quick Start (Windows Users) ⚡

**Install dependencies:**
```bash
pip install ultralytics opencv-python numpy
```

**Run detection:**
```bash
python detect.py your_image.jpg
```

**Output:**
- Console: Detection results with confidence scores
- File: `your_image_detected.jpg` with annotated bounding boxes

**Done!** Your annotated image is ready.

---

## Files Overview

| File | Type | Purpose | Use Case |
|------|------|---------|----------|
| `detect.py` | Python | Simple all-in-one detector | **Recommended for Windows/Testing** |
| `ipc_yolov4.c` | C | IPC detector (Darknet) | Embedded systems with Darknet |
| `ipc_yolov4_replica.py` | Python | IPC detector (YOLOv5) | Testing IPC without C compilation |
| `ipc_reader.py` | Python | IPC reader & renderer | Reads detections from shared memory |

---

## Approach 1: Simple Detection (Recommended)

### What It Does
`detect.py` performs complete object detection in a single script:
1. Loads your image
2. Runs YOLOv5s detection
3. Draws bounding boxes with labels
4. Saves annotated output

### Usage
```bash
# Basic usage
python detect.py image.jpg

# Custom confidence threshold (default: 0.5)
python detect.py image.jpg 0.7

# Custom confidence and NMS threshold
python detect.py image.jpg 0.6 0.4
```

### Example Output
```
Image: dog.jpeg (640x480)
Loading YOLOv5s model...
Running detection (conf=0.5, iou=0.45)...

Detected 3 object(s):
  [0] dog: conf=0.941 box=(125,180,320,280)
  [1] person: conf=0.876 box=(400,50,180,420)
  [2] car: conf=0.732 box=(10,300,250,150)

✓ Saved annotated image to: dog_detected.jpeg
```

---

## Approach 2: IPC Detection (Advanced)

### Architecture
```
┌──────────────────────────────────────┐
│ DETECTOR PROCESS                     │
│ (ipc_yolov4_replica.py or .c)        │
│                                      │
│ 1. Load image                        │
│ 2. Run YOLOv5/YOLOv4 detection      │
│ 3. Write results to shared memory    │
└──────────────────────────────────────┘
                ↓
    /dev/shm/ipc_yolov4_shm
    (Shared Memory: 264 bytes)
                ↓
┌──────────────────────────────────────┐
│ READER PROCESS                       │
│ (ipc_reader.py)                      │
│                                      │
│ 1. Read detections from memory       │
│ 2. Draw bounding boxes on image      │
│ 3. Save annotated output             │
└──────────────────────────────────────┘
```

### How It Works Together

**Step 1: Detector runs and populates shared memory**
```python
# ipc_yolov4_replica.py does:
1. Detects objects using YOLOv5
2. Creates shared memory buffer (/dev/shm/ipc_yolov4_shm)
3. Writes detection results:
   - count: number of objects found
   - detections[]: array of {class_id, confidence, x, y, w, h}
```

**Step 2: Reader accesses shared memory**
```python
# ipc_reader.py does:
1. Opens shared memory buffer (same name)
2. Reads detection count and bounding boxes
3. Loads original image
4. Draws boxes and labels
5. Saves as <image>_detected.<ext>
```

### Shared Memory Data Structure

```c
// Maximum 10 detections per image
#define MAX_BOXES 10

typedef struct {
    int class_id;      // Object class (0=person, 2=car, etc.)
    float confidence;  // Detection confidence (0.0-1.0)
    int x, y;         // Top-left corner of bounding box
    int w, h;         // Width and height of box
} Detection;

typedef struct {
    int count;              // How many objects detected
    Detection det[10];      // Array of detections
} SharedData;              // Total: ~264 bytes
```

### Usage (Linux/WSL)

**Option A: Python Replica (No C compilation)**
```bash
# Terminal 1: Run detector
python ipc_yolov4_replica.py sample.png

# Terminal 2: Read and render
python ipc_reader.py sample.png
```

**Option B: C Program (Requires Darknet)**
```bash
# Compile C program
gcc -o ipc_yolov4 ipc_yolov4.c -ldarknet -lm

# Terminal 1: Run C detector
./ipc_yolov4 sample.jpg

# Terminal 2: Read and render
python ipc_reader.py sample.jpg
```

### Why Use IPC Approach?

**Benefits:**
- **Modularity**: Detector and renderer run as separate processes
- **Language Flexibility**: C detector + Python renderer (best of both worlds)
- **Real-time**: Continuous detection stream possible
- **Lightweight**: Only detection data shared, not full images
- **Production-ready**: Mirrors embedded system architecture

**When to Use:**
- Embedded systems with C/C++ detection engine
- Multi-process architecture requirements
- Learning IPC concepts for systems programming
- Production deployment on Linux/Unix systems

---

## Comparison: Simple vs IPC

| Feature | Simple (`detect.py`) | IPC (replica + reader) |
|---------|---------------------|------------------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ One command | ⭐⭐⭐ Two processes |
| **Windows Support** | ✅ Native | ⚠️ WSL/Linux only |
| **Setup** | Just run it | Shared memory required |
| **Speed** | Fast (direct) | Fast (IPC overhead minimal) |
| **Use Case** | Testing, development | Production, embedded |
| **Processes** | Single | Multiple (detector + reader) |
| **Best For** | Quick results | System architecture |

---

## Detection Parameters

Both approaches support:
- **Confidence Threshold**: 0.5 (50% minimum confidence)
- **NMS Threshold**: 0.45 (Non-Maximum Suppression for overlapping boxes)
- **Max Detections**: 10 objects per image (IPC limit, unlimited for simple)
- **Model**: YOLOv5s (7MB, 80 object classes from COCO dataset)

### COCO Classes Detected
Common objects include: person, car, dog, cat, bicycle, motorcycle, bus, truck, bird, boat, and 70 more.

---

## Performance

| Metric | Time | Notes |
|--------|------|-------|
| Model Download | ~10s | First run only, cached afterward |
| Model Load | ~1-2s | Each run |
| Detection (YOLOv5s) | ~0.1-0.5s | Depends on image size |
| Rendering | ~0.05s | Drawing boxes |
| Shared Memory I/O | <1ms | IPC approach only |

**Total**: ~2-3 seconds per image (first run), ~0.5-1s subsequent runs

---

## Dependencies

### Simple Approach (`detect.py`)
```bash
pip install ultralytics opencv-python numpy
```
- **ultralytics**: YOLOv5 model and inference
- **opencv-python**: Image I/O and rendering
- **numpy**: Array operations

### IPC Approach (Python Replica)
Same as simple approach (works on Linux/WSL)

### IPC Approach (C Program)
- **Darknet**: Full YOLOv4 framework
- **GCC/Clang**: C compiler
- **POSIX OS**: Linux, macOS, or WSL2

---

## Troubleshooting

### "Image not found"
- Check file path and extension
- Use absolute path or ensure file is in current directory

### "Shared memory not available" (IPC)
- **Windows**: Use WSL2 or Docker for Linux environment
- **Alternative**: Use simple `detect.py` instead

### "Model download fails"
- Check internet connection
- Models download from GitHub (requires access)
- Default cache: `~/.cache/ultralytics/`

### "No objects detected"
- Lower confidence threshold: `python detect.py image.jpg 0.3`
- Check image quality and object visibility
- Try different image with clear objects

### Low detection accuracy
- Objects too small or unclear
- Try YOLOv5m for better accuracy: Edit code to use `'yolov5m.pt'`
- Increase image resolution

---

## Advanced Usage

### Process Flow Diagram (IPC)

```
Time →

T0: Detector starts
    ├─ Load YOLOv5 model (1s)
    └─ Load image

T1: Detection runs (0.5s)
    └─ Inference on image

T2: Write to shared memory (<1ms)
    ├─ Create /dev/shm/ipc_yolov4_shm
    ├─ Write count=3
    └─ Write 3 Detection structs

T3: Detector exits

T4: Reader starts (can run anytime after T2)
    ├─ Open shared memory
    └─ Read SharedData struct

T5: Render (0.1s)
    ├─ Load original image
    ├─ Draw 3 bounding boxes
    └─ Save output

T6: Done ✓
```

### Customizing Detection

**Edit confidence in `detect.py`:**
```python
# Line ~68
def detect_and_render(image_path, output_path=None, conf=0.5, iou=0.45):
    # Change conf=0.5 to conf=0.3 for more detections
```

**Edit model in any script:**
```python
# Change from YOLOv5s to YOLOv5m (better accuracy, slower)
model = YOLO('yolov5m.pt')  # instead of 'yolov5s.pt'
```

---

## Project Structure Summary

```
EmbeddedAssignment/
├── detect.py                    # ⭐ Simple detector (USE THIS)
├── ipc_yolov4_replica.py       # IPC detector (Python)
├── ipc_reader.py               # IPC reader (Python)
├── ipc_yolov4.c                # IPC detector (C/Darknet)
├── README.md                    # This file
├── sample.png                   # Test image (if present)
└── sample_detected.png          # Output (after running)
```

**Recommended workflow:**
1. Start with `python detect.py your_image.jpg`
2. View `your_image_detected.jpg`
3. Explore IPC approach for learning/production use

---

## Summary

✅ **For quick object detection**: Use `detect.py`  
✅ **For embedded systems/IPC learning**: Use IPC approach  
✅ **Windows users**: Stick with `detect.py`  
✅ **Linux/WSL users**: Try both approaches

Both implementations produce the same quality results - choose based on your needs!
