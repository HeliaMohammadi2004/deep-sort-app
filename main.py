"""
main.py
========

This script provides a simple graphical user interface (GUI) for running
Deep‑SORT object tracking on video files or a webcam stream.  The GUI is
implemented using Python's built‑in ``tkinter`` library and therefore does not
require any external dependencies beyond those already specified in
``requirements.txt``.  If ``tkinter`` is not installed on your system, you
may need to install the ``python3‑tk`` package via your package manager (e.g.
``sudo apt install python3‑tk`` on Debian/Ubuntu).

The core tracking logic is encapsulated in the :mod:`tracking` module.

Usage
-----

Run the script from the command line::

    python3 main.py

You will be presented with a window containing buttons to select a video file,
start the tracking process, stop it, and exit the application.  If you do not
select a file, the application will default to using your computer's primary
webcam (source index 0).  During tracking, each frame is displayed in the
window along with bounding boxes and unique IDs for each tracked object.

Tracking runs in a separate thread to keep the UI responsive.  The ``Stop``
button sets a flag that gracefully stops the processing loop after the current
frame.  Pressing the ``Quit`` button will stop any ongoing tracking and close
the application.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

from tracking import ObjectTracker


class DeepSortApp:
    """A minimal Tkinter application for Deep‑SORT object tracking."""

    def __init__(self) -> None:

        self.root = tk.Tk()
        self.root.title("Deep SORT Object Tracking")


        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.TOP, fill=tk.X)


        self.btn_select = tk.Button(
            btn_frame,
            text="Select Video",
            command=self.select_video,
            width=15,
        )
        self.btn_select.pack(side=tk.LEFT, padx=5, pady=5)


        self.btn_start = tk.Button(
            btn_frame,
            text="Start",
            command=self.start,
            state=tk.DISABLED,
            width=10,
        )
        self.btn_start.pack(side=tk.LEFT, padx=5, pady=5)


        self.btn_stop = tk.Button(
            btn_frame,
            text="Stop",
            command=self.stop,
            state=tk.DISABLED,
            width=10,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5, pady=5)


        self.btn_quit = tk.Button(
            btn_frame,
            text="Quit",
            command=self.quit,
            width=10,
        )
        self.btn_quit.pack(side=tk.LEFT, padx=5, pady=5)


        self.canvas = tk.Label(self.root)
        self.canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


        self.video_source: Optional[str | int] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.tracker = ObjectTracker()

    def select_video(self) -> None:
        """Open a file dialog to select a video file for tracking."""
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_source = path
            self.btn_start.config(state=tk.NORMAL)
        else:
            self.video_source = 0
            self.btn_start.config(state=tk.NORMAL)

    def start(self) -> None:
        """Start the video capture and tracking process."""
        if self.running:
            return
        if self.video_source is None:
            self.video_source = 0
        self.btn_select.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.running = True
        self.thread = threading.Thread(target=self.process)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        """Stop the video capture and tracking loop."""
        if not self.running:
            return
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_select.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def quit(self) -> None:
        """Terminate the application."""
        self.stop()
        self.root.quit()

    def process(self) -> None:
        """Read frames from the video source, run tracking, and update the UI."""

        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Error", f"Could not open video source: {self.video_source}"
            )
            self.running = False
            return

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            objects = self.tracker.update(frame)

            for obj in objects:
                l, t, r, b = obj["bbox"]
                track_id = obj["track_id"]
                class_id = obj["class_id"]

                color = (
                    int(37 * track_id % 255),
                    int(17 * track_id % 255),
                    int(29 * track_id % 255),
                )
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                label = f"ID:{track_id}"
                if class_id is not None:
                    label += f" | {self.tracker.model.names[class_id]}"
                cv2.putText(
                    frame,
                    label,
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=pil_image)

            self.canvas.imgtk = imgtk

            self.canvas.config(image=imgtk)

            self.root.update_idletasks()
            self.root.update()

            time.sleep(0.01)

        if self.cap is not None:
            self.cap.release()
        self.running = False

        self.btn_select.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def run(self) -> None:
        """Enter the Tkinter main event loop."""
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.mainloop()


if __name__ == "__main__":
    app = DeepSortApp()
    app.run()