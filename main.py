"""
YOLOv8 Chess Piece Detection GUI (Tkinter) - Improved (no flicker, threaded)

Requires:
    pip install ultralytics opencv-python numpy pillow torch
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Chess - Vision (stable)")

        # Model / device
        self.model = None
        self.model_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())

        # Camera + threading
        self.cap = None
        self.cam_index_var = tk.StringVar(value="0")
        self.running = False
        self.capture_thread = None
        self.infer_thread = None
        self.stop_event = threading.Event()

        # Shared frames
        self.raw_frame = None            # latest BGR frame from camera
        self.annotated_frame = None      # latest annotated BGR frame (inference result)
        self.frame_lock = threading.Lock()

        # Inference settings
        self.conf_threshold = tk.DoubleVar(value=0.25)
        self.infer_fps = 0.0
        self.display_fps = 0.0

        # Build UI
        self.build_ui()

        # GUI update loop (uses after)
        self.last_display_time = time.time()
        self.root.after(30, self.gui_loop)  # roughly 33ms -> ~30 FPS UI

    def build_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_model = tk.Button(top_frame, text="Chọn model (.pt)", command=self.select_model)
        btn_model.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_model = tk.Label(top_frame, text="Chưa chọn model")
        self.lbl_model.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Camera index:").pack(side=tk.LEFT, padx=5)
        self.entry_cam = tk.Entry(top_frame, textvariable=self.cam_index_var, width=5)
        self.entry_cam.pack(side=tk.LEFT)

        btn_start = tk.Button(top_frame, text="Start", command=self.toggle_camera)
        btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_start = btn_start

        # GPU checkbox
        self.chk_gpu = tk.Checkbutton(top_frame, text=" GPU (CUDA)", variable=self.use_gpu, command=self.toggle_device)
        self.chk_gpu.pack(side=tk.LEFT, padx=10)
        if not torch.cuda.is_available():
            self.chk_gpu.config(state=tk.DISABLED)

        # Confidence slider
        conf_frame = tk.Frame(self.root)
        conf_frame.pack(side=tk.TOP, fill=tk.X, padx=6)
        tk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
        conf_slider = tk.Scale(conf_frame, variable=self.conf_threshold, from_=0.01, to=0.9, resolution=0.01, orient=tk.HORIZONTAL, length=250)
        conf_slider.pack(side=tk.LEFT, padx=5)

        # Canvas for video
        self.canvas = tk.Label(self.root)
        self.canvas.pack(padx=10, pady=10)

        # Status bar
        status_frame = tk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status = tk.Label(status_frame, text=f"Thiết bị: {self.device} | Ready")
        self.lbl_status.pack(side=tk.LEFT, padx=5)

        self.lbl_fps_display = tk.Label(status_frame, text="Display FPS: 0.0")
        self.lbl_fps_display.pack(side=tk.RIGHT, padx=5)

        self.lbl_fps_infer = tk.Label(status_frame, text="Infer FPS: 0.0")
        self.lbl_fps_infer.pack(side=tk.RIGHT, padx=10)

    def toggle_device(self):
        # Change device selection; inference thread will pick up new device on next run
        self.device = "cuda" if self.use_gpu.get() and torch.cuda.is_available() else "cpu"
        self.lbl_status.config(text=f"Thiết bị: {self.device}")

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.lbl_status.config(text="Loading model...")
            self.root.update_idletasks()
            # Load model (will be used later on chosen device)
            self.model = YOLO(path)
            self.model_path = path
            self.lbl_model.config(text=path.split("/")[-1])
            self.lbl_status.config(text=f"Model loaded (will run on {self.device})")
        except Exception as e:
            messagebox.showerror("Error", f"Không load được model: {e}")
            self.model = None
            self.lbl_status.config(text="Model load failed")

    def toggle_camera(self):
        if not self.running:
            try:
                cam_index = int(self.cam_index_var.get())
            except:
                cam_index = 0
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # use DirectShow on Windows for stability
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Không mở được camera {cam_index}")
                return

            # Optionally set resolution to reduce load
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.running = True
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            # Start inference thread even if model None (it will wait until model exists)
            self.infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.infer_thread.start()

            self.btn_start.config(text="Stop")
            self.lbl_status.config(text=f"Running (device: {self.device})")
        else:
            self.stop_event.set()
            self.running = False
            self.btn_start.config(text="Start")
            # safe release
            if self.capture_thread:
                self.capture_thread.join(timeout=1.0)
            if self.infer_thread:
                self.infer_thread.join(timeout=1.0)
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            self.lbl_status.config(text="Stopped")

    def _capture_loop(self):
        """Continuous capture from camera, write to raw_frame (protected by lock)."""
        last_time = time.time()
        display_count = 0
        while not self.stop_event.is_set() and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.frame_lock:
                self.raw_frame = frame.copy()
            # small sleep to yield CPU (adjustable)
            time.sleep(0.001)
        # end capture loop

    def _inference_loop(self):
        """Run inference on the latest raw_frame (non-blocking GUI)."""
        frame_count = 0
        prev_time = time.time()
        while not self.stop_event.is_set():
            # Wait for a frame
            with self.frame_lock:
                local_frame = None if self.raw_frame is None else self.raw_frame.copy()

            if local_frame is None:
                time.sleep(0.01)
                continue

            if self.model is None:
                # no model; keep annotated_frame equal to raw frame for display
                with self.frame_lock:
                    self.annotated_frame = local_frame
                time.sleep(0.01)
                continue

            try:
                # Run prediction on the chosen device with current confidence
                t0 = time.time()
                results = self.model.predict(
                    source=local_frame,
                    conf=float(self.conf_threshold.get()),
                    imgsz=640,
                    device=self.device,
                    verbose=False
                )
                t1 = time.time()
                # Annotate copy
                annotated = local_frame.copy()
                r = results[0]
                boxes = getattr(r, "boxes", [])
                for box in boxes:
                    # box.xyxy is a tensor on cpu or device; ensure get to cpu numpy
                    xyxy = box.xyxy.cpu().numpy().astype(int).reshape(-1)
                    x1, y1, x2, y2 = xyxy[:4]
                    conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else 0.0
                    cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else 0
                    name = r.names[cls] if r.names is not None and cls in r.names else str(cls)
                    label = f"{name} {conf:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, label, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # update annotated_frame
                with self.frame_lock:
                    self.annotated_frame = annotated
                # update infer fps
                frame_count += 1
                if frame_count >= 5:
                    now = time.time()
                    self.infer_fps = frame_count / (now - prev_time)
                    prev_time = now
                    frame_count = 0
            except Exception as e:
                # If inference fails, put raw frame to annotated to avoid blank display
                with self.frame_lock:
                    self.annotated_frame = local_frame
                # Show error in status but keep running
                self.lbl_status.config(text=f"Lỗi inference: {e}")
                time.sleep(0.2)

        # end inference loop

    def gui_loop(self):
        """Update tkinter image from annotated_frame (or raw_frame) — runs in main thread."""
        now = time.time()
        img_to_show = None
        with self.frame_lock:
            if self.annotated_frame is not None:
                img_to_show = self.annotated_frame.copy()
            elif self.raw_frame is not None:
                img_to_show = self.raw_frame.copy()

        if img_to_show is not None:
            # Convert BGR -> RGB -> ImageTk
            rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=pil)
            # Avoid flicker: keep reference and set only image data
            self.canvas.imgtk = imgtk
            self.canvas.config(image=imgtk)

            # update display fps
            dt = now - self.last_display_time if self.last_display_time else 0.001
            self.display_fps = 1.0 / dt if dt > 0 else 0.0
            self.last_display_time = now

            self.lbl_fps_display.config(text=f"Display FPS: {self.display_fps:.1f}")
            self.lbl_fps_infer.config(text=f"Infer FPS: {self.infer_fps:.1f}")

        # schedule next update
        self.root.after(30, self.gui_loop)

    def on_closing(self):
        # Graceful shutdown
        if self.running:
            self.stop_event.set()
            self.running = False
            if self.capture_thread:
                self.capture_thread.join(timeout=1.0)
            if self.infer_thread:
                self.infer_thread.join(timeout=1.0)
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
