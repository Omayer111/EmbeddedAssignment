#!/usr/bin/env python3
"""
IPC Reader - Reads detections from shared memory and renders output
This reads detections written by ipc_yolov4.c (or ipc_yolov4_replica.py)
and draws bounding boxes on the image.
"""

import mmap
import ctypes
import cv2
import os
import sys
from ctypes import Structure, c_int, c_float

SHM_NAME = "/ipc_yolov4_shm"
MAX_BOXES = 10


class Detection(Structure):
    _fields_ = [
        ("class_id", c_int),
        ("confidence", c_float),
        ("x", c_int),
        ("y", c_int),
        ("w", c_int),
        ("h", c_int),
    ]


class SharedData(Structure):
    _fields_ = [
        ("count", c_int),
        ("det", Detection * MAX_BOXES),
    ]


def read_shared_memory():
    """
    Read detections from shared memory
    Returns list of detections
    """
    try:
        shm_path = "/dev/shm" + SHM_NAME
        if not os.path.exists(shm_path):
            print(f"Error: Shared memory not found at {shm_path}")
            return None

        fd = os.open(shm_path, os.O_RDONLY)
        shm = mmap.mmap(fd, ctypes.sizeof(SharedData), mmap.MAP_SHARED, mmap.PROT_READ)

        shared = SharedData.from_buffer_copy(shm[:])

        detections = []
        for i in range(shared.count):
            det = shared.det[i]
            detections.append({
                'class_id': det.class_id,
                'confidence': det.confidence,
                'x': det.x,
                'y': det.y,
                'w': det.w,
                'h': det.h,
            })

        shm.close()
        os.close(fd)

        return detections

    except Exception as e:
        print(f"Error reading shared memory: {e}")
        return None


def render_detections(image_path, detections, output_path=None):
    """
    Draw detection boxes on image and save
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_detected{ext}"

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None

        print(f"Rendering {len(detections)} detection(s)...")

        for i, det in enumerate(detections):
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            conf = det['confidence']

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            label = f"ID:{det['class_id']} {conf:.2f}"
            cv2.putText(img, label, (x, max(10, y - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(output_path, img)
        print(f"Saved to {output_path}")
        return output_path

    except Exception as e:
        print(f"Error rendering: {e}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python ipc_reader.py <image_path>")
        print("Example: python ipc_reader.py sample.ppm")
        print("\nNote: Run ipc_yolov4_replica.py first to populate shared memory")
        return 1

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return 1

    # Read from shared memory
    detections = read_shared_memory()
    if detections is None:
        return 1

    print(f"Read {len(detections)} detection(s) from shared memory:")
    for i, det in enumerate(detections):
        print(f"  [{i}] class_id={det['class_id']} conf={det['confidence']:.2f} "
              f"box=({det['x']},{det['y']},{det['w']},{det['h']})")

    # Render on image
    render_detections(image_path, detections)

    return 0


if __name__ == "__main__":
    sys.exit(main())
