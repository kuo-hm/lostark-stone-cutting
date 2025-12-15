import mss
import cv2
import numpy as np
from PIL import Image
import threading
import time
import subprocess
import shutil

try:
    from ewmh import EWMH
except ImportError:
    EWMH = None

class ScreenCapture:
    """
    Handles screen capture functionality using MSS for general capture
    and Grim for Wayland support. It runs capturing in a separate thread.
    """
    def __init__(self):
        self.ewmh = EWMH() if EWMH else None
        self.running = False
        self.target_window_id = None
        self.capture_region = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.use_grim = False

    def get_windows(self):
        """
        Returns a list of select-able windows (id, name) using EWMH.
        """
        if not self.ewmh:
            return []
        
        windows = []
        try:
            client_list = self.ewmh.getClientList()
            for win in client_list:
                name = win.get_wm_name()
                if name:
                    windows.append({"id": win.id, "name": name})
        except Exception as e:
            print(f"Error listing windows: {e}")
        return windows

    def set_target_window(self, window_id):
        """
        Sets the target window ID for tracking and updates the initial capture region.
        """
        self.target_window_id = window_id
        self.update_region_from_window()

    def update_region_from_window(self):
        """
        Updates the capture region coordinates based on the current position 
        and size of the target window.
        """
        if not self.ewmh or not self.target_window_id:
            return

        try:
            win = None
            for w in self.ewmh.getClientList():
                if w.id == self.target_window_id:
                    win = w
                    break
            
            if win:
                geo = win.get_geometry()
                self.capture_region = {
                    "top": geo.y,
                    "left": geo.x,
                    "width": geo.width,
                    "height": geo.height
                }
                print(f"Tracking window '{win.get_wm_name()}' at {self.capture_region}")
        except Exception as e:
            print(f"Error updating region: {e}")

    def start(self):
        """
        Starts the background capture thread.
        """
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """
        Stops the background capture thread.
        """
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

    def _capture_loop(self):
        """
        Main loop for capturing screen frames. Handles switching between MSS and Grim
        if necessary (e.g., on Wayland) and manages frame rate.
        """
        with mss.mss() as sct:
            while self.running:
                if self.target_window_id:
                    # Periodically check window position
                    if time.time() % 2 < 0.1:
                         self.update_region_from_window()
                
                region = self.capture_region
                if not region:
                    region = sct.monitors[1]

                try:
                    img = None
                    
                    if self.use_grim:
                        img = self._capture_grim(region)
                        if img is None:
                            time.sleep(1)
                            continue
                    else:
                        try:
                            img_mss = sct.grab(region)
                            img = np.array(img_mss)
                        except Exception as e:
                            print(f"MSS Capture error: {e}")
                            if "XGetImage" in str(e) and shutil.which("grim"):
                                print("Switching to grim for Wayland capture...")
                                self.use_grim = True
                                continue
                            else:
                                time.sleep(1)
                                continue

                    if img is not None:
                        with self.lock:
                            self.latest_frame = img
                    
                except Exception as e:
                    print(f"Loop error: {e}")
                    time.sleep(1)
                
                time.sleep(1/10 if self.use_grim else 1/30)

    def _capture_grim(self, region):
        """
        Captures a screen region using the 'grim' utility (Wayland support).
        Returns the image as a BGRA numpy array.
        """
        try:
            x = int(region['left'])
            y = int(region['top'])
            w = int(region['width'])
            h = int(region['height'])
            
            geom = f"{x},{y} {w}x{h}"
            cmd = ["grim", "-g", geom, "-t", "ppm", "-"]
            
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                return None
                
            buff = np.frombuffer(res.stdout, dtype=np.uint8)
            frame = cv2.imdecode(buff, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Convert BGR to BGRA to match MSS format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                
            return frame

        except Exception as e:
            print(f"Grim error: {e}")
            return None

    def get_latest_frame(self):
        """
        Thread-safe retrieval of the latest captured frame (numpy array).
        """
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def get_latest_frame_as_image(self):
        """
        Returns the latest frame as a PIL Image converted to RGB, suitable for UI display.
        """
        frame = self.get_latest_frame()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(frame)
        return None

    def find_template(self, template_path, threshold=0.8):
        """
        Searches for a template image within the current screen frame.
        Returns a dictionary with 'rect', 'center', and 'confidence' if found, else None.
        """
        current_frame = self.get_latest_frame()
        if current_frame is None:
            print("No frame to search in")
            return None

        try:
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                print(f"Failed to load template: {template_path}")
                return None

            # Convert frame to BGR for matching
            frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
            
            result = cv2.matchTemplate(frame_bgr, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                h, w = template.shape[:2]
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                center = (top_left[0] + w//2, top_left[1] + h//2)
                
                print(f"Found template at {top_left} with confidence {max_val:.2f}")
                return {
                    "rect": (top_left, bottom_right),
                    "center": center,
                    "confidence": max_val
                }
            else:
                print(f"Template not found. Max confidence: {max_val:.2f}")
                return None
                
        except Exception as e:
            print(f"Error in find_template: {e}")
            return None
