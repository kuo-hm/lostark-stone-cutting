import customtkinter as ctk
import tkinter as tk
from PIL import Image
from vision_service import ScreenCapture
from stone_logic import calculate_best_move, GoalConfig
from gemini_service import GeminiAdvisor

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class Diamond(ctk.CTkCanvas):
    """
    A custom canvas widget representing a single ability stone node (diamond).
    Supports states: empty, success, fail.
    """
    def __init__(self, master, size=30, success_color="#3b82f6", **kwargs):
        super().__init__(master, width=size, height=size, highlightthickness=0, **kwargs)
        self.size = size
        self.success_color = success_color
        
        # Resolve background color to match parent
        bg_color = master.cget("fg_color")
        if bg_color == "transparent":
            try:
                bg_color = master.master.cget("fg_color")
            except Exception:
                pass
                
        if isinstance(bg_color, (list, tuple)):
            mode = ctk.get_appearance_mode()
            if mode == "Light":
                bg_color = bg_color[0]
            else:
                bg_color = bg_color[1]
                
        self.configure(bg=bg_color)
        
        padding = 4
        mid = size / 2
        self.coords = [
            mid, padding,             # Top
            size - padding, mid,      # Right
            mid, size - padding,      # Bottom
            padding, mid              # Left
        ]
        
        self.state = "empty" 
        self.draw()

    def set_state(self, state):
        """Updates the state of the diamond and redraws it."""
        self.state = state
        self.draw()

    def draw(self):
        """Draws the diamond polygon based on current state."""
        self.delete("all")
        color = "#e5e7eb"
        
        if self.state == "success":
            color = self.success_color
        elif self.state == "fail":
            color = "#6b7280" # Gray
        elif self.state == "empty":
            color = "#e5e7eb" 

        self.create_polygon(self.coords, fill=color, outline="")

class StoneRow(ctk.CTkFrame):
    """
    Represents a full row of ability stone nodes (Icon + 10 Diamonds).
    """
    def __init__(self, master, title="Ability", color="blue", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        self.icon = ctk.CTkButton(self, 
                                  text="", 
                                  width=50, 
                                  height=50, 
                                  corner_radius=25, 
                                  fg_color="black", 
                                  border_width=2, 
                                  border_color=color,
                                  hover=False,
                                  state="disabled")
        self.icon.pack(side="left", padx=(0, 15))
        
        self.diamonds = []
        for i in range(10):
            d = Diamond(self, size=30, success_color=color)
            d.pack(side="left", padx=2)
            self.diamonds.append(d)

    def set_icon(self, img_bgr):
        """Updates the circular icon with a new image (BGR numpy array)."""
        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        h_target = 50
        w_target = int((img_pil.width / img_pil.height) * h_target)
        img_pil = img_pil.resize((w_target, h_target), Image.Resampling.LANCZOS)
        
        ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(w_target, h_target))
        self.icon.configure(image=ctk_img, text="", border_width=1)

    def set_diamond_state(self, index, state):
        """Sets the state of a specific diamond by index."""
        if 0 <= index < len(self.diamonds):
            self.diamonds[index].set_state(state)

class StoneCutterApp(ctk.CTk):
    """
    Main Application class for the Lost Ark Stone Cutter tool.
    Handles GUI layout, interaction, and coordination with the Vision Service.
    """
    def __init__(self):
        super().__init__()
        
        self.vision = ScreenCapture()
        self.gemini = GeminiAdvisor()
        
        # Load GUI
        # self.vision.start() # No continuous capture

        self.title("Lost Ark Stone Cutter")
        self.geometry("800x600")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.render_header()
        self.render_preview_area()
        self.render_status()
        self.render_stones()

    def render_header(self):
        """Renders the top header section."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        # Left side buttons
        # 77 Mode: Default
        self.btn_77 = ctk.CTkButton(header_frame, text="77 Mode", width=100, fg_color="#f97316", text_color="black", hover_color="#fdba74", border_width=2, border_color="#f97316", command=lambda: self.set_mode("77")) # Selected style
        self.btn_77.pack(side="left", padx=5)
        
        # 97 Mode
        self.btn_97 = ctk.CTkButton(header_frame, text="97 Mode", width=100, fg_color="white", text_color="gray", hover_color="#f3f4f6", border_width=1, border_color="#e5e7eb", command=lambda: self.set_mode("97"))
        self.btn_97.pack(side="left", padx=5)
        
        self.target_line_var = ctk.StringVar(value="Line 1")
        self.line_selector = ctk.CTkComboBox(header_frame, values=["Line 1", "Line 2"], variable=self.target_line_var, width=100, state="readonly")
        # Don't pack initially (Default 77 mode)

        self.target_mode = "77" # Default
        
        # Calc Mode Switch
        self.calc_mode_var = ctk.StringVar(value="Algo")
        self.calc_switch = ctk.CTkSegmentedButton(header_frame, values=["Algo", "Gemini"], variable=self.calc_mode_var, command=self.on_calc_mode_change)
        self.calc_switch.pack(side="left", padx=20)

        monitor_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        monitor_frame.pack(side="right")
        
        btn_monitor = ctk.CTkButton(monitor_frame, text="Select Monitor", fg_color="white", text_color="black", border_width=2, border_color="black", hover_color="#f3f4f6", command=self.open_monitor_selection)
        btn_monitor.pack(side="left", padx=5)

    def render_status(self):
        """Renders the status button area."""
        self.status_frame = ctk.CTkFrame(self, fg_color="#f9fafb", corner_radius=10)
        self.status_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        # Create a sub-frame for buttons to keep them grouped
        button_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        button_frame.pack(pady=(20, 10)) # Pack this frame with padding
        
        self.status_btn = ctk.CTkButton(button_frame, text="Scan & Calculate", width=200, height=40, font=("Inter", 16, "bold"), fg_color="white", text_color="#374151", hover_color="#f3f4f6", command=self.perform_recognition)
        self.status_btn.pack(side="left", padx=10)
        
        self.status_btn.pack(side="left", padx=10)
        
        # Calculate Button removed (merged)

        self.prob_label = ctk.CTkLabel(self.status_frame, text="Current Chance: 75%", font=("Inter", 16, "bold"), text_color="#3b82f6")
        self.prob_label.pack(pady=(0, 5))
        
        self.goal_status_label = ctk.CTkLabel(self.status_frame, text="Goal: Pending", font=("Inter", 14), text_color="gray")
        self.goal_status_label.pack(pady=(0, 20))

    def render_stones(self):
        """Renders the main ability stone rows."""
        container = ctk.CTkFrame(self)
        container.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        
        self.ability1 = StoneRow(container, title="Ability 1", color="#3b82f6")
        self.ability1.pack(pady=20)
        
        self.ability2 = StoneRow(container, title="Ability 2", color="#3b82f6")
        self.ability2.pack(pady=20)
        
        self.malice = StoneRow(container, title="Malice", color="#ef4444")
        self.malice.pack(pady=20)

    def render_preview_area(self):
        """Renders the live preview area."""
        self.preview_frame = ctk.CTkFrame(self, fg_color="#1f2937", corner_radius=10)
        self.preview_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="[Live Preview]", text_color="gray")
        self.preview_label.pack(fill="both", expand=True, padx=2, pady=2)
        
        self.update_preview()

    def set_mode(self, mode):
        self.target_mode = mode
        if mode == "77":
            self.btn_77.configure(fg_color="#f97316", border_width=2, text_color="black")
            self.btn_97.configure(fg_color="white", border_width=1, text_color="gray")
            self.line_selector.pack_forget()
        else:
            self.btn_77.configure(fg_color="white", border_width=1, text_color="gray")
            self.btn_97.configure(fg_color="#f97316", border_width=2, text_color="black")
            # Pack after btn_97
            self.line_selector.pack(side="left", padx=5, after=self.btn_97)
        
    def on_calc_mode_change(self, value):
        print(f"Logic Mode changed to: {value}")
            
    def get_row_stats(self, row):
        success = 0
        fail = 0
        remaining = 0
        for d in row.diamonds:
            if d.state == "success": success += 1
            elif d.state == "fail": fail += 1
            elif d.state == "empty": remaining += 1
        return success, fail, remaining

    def run_optimization(self):
        # 1. Gather State
        try:
            # Get probability from label "Current Chance: XX%"
            txt = self.prob_label.cget("text")
            import re
            match = re.search(r'(\d+)%', txt)
            if match:
                prob_val = int(match.group(1)) / 100.0
            else:
                prob_val = 0.75
        except:
            prob_val = 0.75

        # 3. Gather State (Common for both Algo and Gemini)
        l1_s, l1_f, l1_r = self.get_row_stats(self.ability1)
        l2_s, l2_f, l2_r = self.get_row_stats(self.ability2)
        m_s, m_f, m_r = self.get_row_stats(self.malice)
        
        state = {
            "prob": prob_val,
            "l1_s": l1_s, "l1_r": l1_r,
            "l2_s": l2_s, "l2_r": l2_r,
            "mal_s": m_s, "mal_r": m_r
        }
        
        print(f"Current State: {state}")
        
        # 4. Define Goals
        goals = []
        if self.target_mode == "97":
            target_line = self.target_line_var.get()
            print(f"97 Mode Target: {target_line}")
            
            if target_line == "Line 1":
                # Line 1 needs 9, Line 2 needs 7
                goals.append(GoalConfig(9, 7, 4, strict_order=True))
            else:
                # Line 1 needs 7, Line 2 needs 9
                goals.append(GoalConfig(7, 9, 4, strict_order=True))
                
            # Fallback 7/7
            goals.append(GoalConfig(7, 7, 4, strict_order=True))
        else:
            # 7/7
            goals.append(GoalConfig(7, 7, 4, strict_order=True))
            
        # 5. Run Logic
        best_move = None
        current_mode = self.calc_mode_var.get()
        active_goal = None # Initialize to prevent UnboundLocalError
        
        if current_mode == "Algo":
            best_move, moves, active_goal = calculate_best_move(state, goals)
            print(f"Math Engine Result: {best_move} | Moves: {moves}")
        else:
            # Gemini Logic Mode
            target_line = self.target_line_var.get() if self.target_mode == "97" else "Any"
            print("Sending State to Gemini Logic...")
            move, reason = self.gemini.get_suggestion(state, self.target_mode, target_line)
            if move:
                best_move = move
                print(f"Gemini Result: {best_move} ({reason})")
            else:
                print(f"Gemini Logic Failed: {reason}")
        
        # 6. Check Feasibility (Update Goal Status)
        # We check if the *Primary* goal for the current mode is possible.
        # We can deduce this by running calculate_best_move with ONLY the primary goal.
        
        primary_goal = None
        if self.target_mode == "97":
            t_line = self.target_line_var.get()
            if t_line == "Line 1": primary_goal = GoalConfig(9, 7, 4, strict_order=True)
            else: primary_goal = GoalConfig(7, 9, 4, strict_order=True)
            target_str = "9/7"
        else:
            primary_goal = GoalConfig(7, 7, 4, strict_order=True)
            target_str = "7/7"
            
        check_move, check_probs, check_goal = calculate_best_move(state, [primary_goal])
        
        if check_goal is not None:
             # Get max probability from the checking result
             max_prob = max(check_probs.values()) if check_probs else 0.0
             sub_percent = max_prob * 100
             self.goal_status_label.configure(text=f"{target_str} Possible ({sub_percent:.4f}%)", text_color="#10b981") # Green
        else:
             self.goal_status_label.configure(text=f"{target_str} IMPOSSIBLE", text_color="#ef4444") # Red
        
        # 4. Highlight UI
        self.ability1.configure(border_width=0)
        self.ability2.configure(border_width=0)
        self.malice.configure(border_width=0)
        
        highlight_color = "#10b981" # Emerald
        
        if best_move == "Line 1":
            self.ability1.configure(border_width=3, border_color=highlight_color)
        elif best_move == "Line 2":
            self.ability2.configure(border_width=3, border_color=highlight_color)
        elif best_move == "Malus":
            self.malice.configure(border_width=3, border_color=highlight_color)
            
        # Optional: Show EV/Goal info in status
        if active_goal:
             # self.status_btn.configure(text=f"Target: {active_goal.target_main}/{active_goal.target_off} | Best: {best_move}")
             pass

    def detect_nodes(self, nodes_img, is_malice=False):
        """
        Analyzes the nodes image region to determine the state of each diamond.
        Splits the image into 10 chunks and uses template matching + color validation.
        Returns a list of states (success, fail, empty).
        """
        import cv2
        import numpy as np
        import os
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        folder_name = "nodes/bad" if is_malice else "nodes"
        folder = os.path.join(base_path, folder_name)
        
        t_normal = cv2.imread(os.path.join(folder, "normal.png"))
        t_success = cv2.imread(os.path.join(folder, "success.png"))
        t_failed = cv2.imread(os.path.join(folder, "failed.png"))
        
        if t_normal is None or t_success is None or t_failed is None:
            print("Error loading node templates")
            return ["empty"] * 10
            
        # Ensure BGR format matches templates
        if len(nodes_img.shape) == 3 and nodes_img.shape[2] == 4:
            nodes_img = cv2.cvtColor(nodes_img, cv2.COLOR_BGRA2BGR)
            
        states = []
        h, w = nodes_img.shape[:2]
        chunk_w = w // 10
        
        for i in range(10):
            x_start = i * chunk_w
            x_end = min(x_start + chunk_w, w)
            chunk = nodes_img[:, x_start:x_end]
            
            # Template matching
            scores = {}
            for name, tmpl in [("empty", t_normal), ("success", t_success), ("fail", t_failed)]:
                 th, tw = tmpl.shape[:2]
                 ch, cw = chunk.shape[:2]
                 
                 if th > ch or tw > cw:
                     tmpl = cv2.resize(tmpl, (cw, ch))
                 
                 res = cv2.matchTemplate(chunk, tmpl, cv2.TM_CCOEFF_NORMED)
                 _, max_val, _, _ = cv2.minMaxLoc(res)
                 scores[name] = max_val
            
            best_state = max(scores, key=scores.get)
            best_score = scores[best_state]
            
            print(f"Node {i}: Emp={scores['empty']:.2f} Suc={scores['success']:.2f} Fail={scores['fail']:.2f} -> {best_state}")
            
            # Color Validation
            ch, cw = chunk.shape[:2]
            cy, cx = ch // 2, cw // 2
            center_crop = chunk[cy-5:cy+5, cx-5:cx+5]
            if center_crop.size > 0:
                avg_bgr = np.mean(center_crop, axis=(0, 1))
                b, g, r = avg_bgr
                
                is_color_match = False
                if is_malice:
                    # Red dominance check
                    # Red should be dominant over Blue and Green
                    is_color_match = (r > b + 20) and (r > g + 20)
                    debug_color = f"R={r:.1f} G={g:.1f} B={b:.1f}"
                else:
                    # Blue dominance check
                    is_color_match = (b > r + 20) and (b > g + 20)
                    debug_color = f"B={b:.1f} G={g:.1f} R={r:.1f}"
                
                # Override: Success must match the expected color
                if best_state == "success" and not is_color_match:
                     print(f"Node {i}: Overriding Success -> Fail (Color Mismatch {debug_color})")
                     best_state = "fail"
            
            if best_score < 0.5:
                best_state = "empty"
                
            states.append(best_state)
            
        return states

    def open_monitor_selection(self):
        """Opens a modal dialog to select the target window."""
        top = ctk.CTkToplevel(self)
        top.title("Select Window/Monitor")
        top.geometry("400x500")
        top.attributes("-topmost", True)
        
        label = ctk.CTkLabel(top, text="Select a Window to Capture", font=("Arial", 16, "bold"))
        label.pack(pady=10)
        
        scrollable_frame = ctk.CTkScrollableFrame(top, width=380, height=400)
        scrollable_frame.pack(padx=10, pady=10)
        
        windows = self.vision.get_windows()
        if not windows:
            ctk.CTkLabel(scrollable_frame, text="No windows found (ewmh missing?)").pack()
        
        for win in windows:
            btn = ctk.CTkButton(scrollable_frame, text=win['name'], anchor="w", command=lambda w=win: self.select_window(top, w))
            btn.pack(fill="x", pady=2)

    def select_window(self, top_window, window_data):
        """Callback for window selection."""
        print(f"Selected: {window_data['name']}")
        self.vision.set_target_window(window_data['id'])
        top_window.destroy()

    def calculate_probability(self):
        """
        Calculates the current success probability based on the total successes and fails.
        Base: 75%
        Success: -10%
        Fail: +10%
        Clamped: [25%, 75%]
        
        We sum states from all 3 rows.
        """
        total_success = 0
        total_fail = 0
        
        # Collect from all diamonds
        for row in [self.ability1, self.ability2, self.malice]:
            for d in row.diamonds:
                if d.state == "success":
                    total_success += 1
                elif d.state == "fail":
                    total_fail += 1
                    
        # Calculate
        # Since we don't know the order, we assume the net change is purely additive
        # This is an approximation if the caps were hit frequently in between, 
        # but for a static view, it's the standard interpretation.
        base_p = 75
        current_p = base_p + (total_fail * 10) - (total_success * 10)
        
        # Clamp
        final_p = max(25, min(75, current_p))
        
        # Update UI
        self.prob_label.configure(text=f"Current Chance: {final_p}%")
        
        # Color code the probability
        color = "#3b82f6" # Blue
        if final_p <= 35:
            color = "#ef4444" # Red
        elif final_p <= 55:
            color = "#f59e0b" # Orange/Amber
            
        self.prob_label.configure(text_color=color)

    def perform_recognition(self):
        """
        Captures screen, finds anchor, extracts nodes, calculates prob, and runs optimization.
        """
        print("Starting recognition...")
        self.status_btn.configure(text="Scanning...", text_color="orange")
        self.update_idletasks()
        
        # 1. Capture Frame (On-Demand)
        current_frame = self.vision.capture_screenshot()
        
        if current_frame is None:
             print("Screenshot failed.")
             self.status_btn.configure(text="Capture Failed", text_color="red")
             return

        # Update Preview with this static frame
        self._show_static_preview(current_frame)
        
        ocr_active = False # Initialize result
        
        import os
        import cv2
        import numpy as np
        import os
        import pytesseract
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        anchor_path = os.path.join(base_path, "ability-improve.png")
        size_ref_path = os.path.join(base_path, "first_ability.png")
        
        msg = ""
        color = "#374151"
        anchor_res = self.vision.find_template(anchor_path)
        
        if anchor_res:
            print(f"Anchor found at {anchor_res['rect'][0]}")
            
            ref_img = cv2.imread(size_ref_path)
            if ref_img is None:
                print("Could not load reference image for size.")
                row_w, row_h = 740, 50
            else:
                row_h, row_w = ref_img.shape[:2]

            anchor_x, anchor_y = anchor_res['rect'][0] 
            
            y_offset_1 = 61
            y_offset_2 = 152
            y_offset_3 = 277 
            
            padding = 2
            capture_h = row_h + (padding * 2)

            r1_x = anchor_x
            r1_y = anchor_y + y_offset_1 - padding
            
            r2_x = anchor_x
            r2_y = anchor_y + y_offset_2 - padding
            
            r3_x = anchor_x
            r3_y = anchor_y + y_offset_3 - padding
            
            # current_frame is already captured at the start of the function
            if current_frame is not None:
                frame_h, frame_w = current_frame.shape[:2]
                
                # ROW 1
                if r1_y + capture_h <= frame_h and r1_x + row_w <= frame_w:
                    row1_img = current_frame[r1_y:r1_y+capture_h, r1_x:r1_x+row_w].copy()
                    
                    r1_engraving = row1_img[:, 0:90]
                    r1_nodes = row1_img[:, 100:500]
                    
                    # cv2.imwrite(os.path.join(base_path, "debug_row1_engraving.png"), r1_engraving)
                    # cv2.imwrite(os.path.join(base_path, "debug_row1_nodes.png"), r1_nodes)
                    
                    self.ability1.set_icon(r1_engraving)
                    
                    states = self.detect_nodes(r1_nodes)
                    print(f"Row 1 States: {states}")
                    for i, s in enumerate(states):
                        self.ability1.set_diamond_state(i, s)
                    
                    msg += "R1 Saved. "
                else:
                    msg += "R1 OOB. "
                    
                # ROW 2
                if r2_y + capture_h <= frame_h and r2_x + row_w <= frame_w:
                    row2_img = current_frame[r2_y:r2_y+capture_h, r2_x:r2_x+row_w].copy()
                    
                    r2_engraving = row2_img[:, 0:90]
                    r2_nodes = row2_img[:, 100:500]
 
                    # cv2.imwrite(os.path.join(base_path, "debug_row2_engraving.png"), r2_engraving)
                    # cv2.imwrite(os.path.join(base_path, "debug_row2_nodes.png"), r2_nodes)
                    
                    self.ability2.set_icon(r2_engraving)
                    
                    states = self.detect_nodes(r2_nodes)
                    print(f"Row 2 States: {states}")
                    for i, s in enumerate(states):
                        self.ability2.set_diamond_state(i, s)
                    
                    msg += "R2 Saved."
                else:
                    msg += "R2 OOB."
                
                # ROW 3
                if r3_y + capture_h <= frame_h and r3_x + row_w <= frame_w:
                    row3_img = current_frame[r3_y:r3_y+capture_h, r3_x:r3_x+row_w].copy()
                    
                    r3_engraving = row3_img[:, 0:90]
                    r3_nodes = row3_img[:, 100:500]

                    # cv2.imwrite(os.path.join(base_path, "debug_row3_engraving.png"), r3_engraving)
                    # cv2.imwrite(os.path.join(base_path, "debug_row3_nodes.png"), r3_nodes)
                    
                    self.malice.set_icon(r3_engraving)
                    
                    # Malice Recognition
                    states = self.detect_nodes(r3_nodes, is_malice=True)
                    print(f"Malice States: {states}")
                    for i, s in enumerate(states):
                        self.malice.set_diamond_state(i, s)
                    
                    msg += " M Saved."
                else:
                    msg += " M OOB."

                color = "green"
            else:
                msg = "No Frame"
                color = "red"
                
            # OCR Success Rate
            # Search for success_rate.png
            success_template_path = os.path.join(base_path, "success_rate.png")
            rate_res = self.vision.find_template(success_template_path, threshold=0.9)
            
            ocr_active = False
            if rate_res:
                print(f"Success Rate Anchor found at {rate_res['rect']}")
                # Extract ROI +5px right
                rt_x, rt_y = rate_res['rect'][0]
                rt_w = rate_res['rect'][1][0] - rt_x
                rt_h = rate_res['rect'][1][1] - rt_y
                
                roi_x = rt_x + rt_w -2 # Shift left by 2px (was +5) to avoid cutting numbers
                roi_y = rt_y
                roi_w = 50 # Capture enough width for "75%"
                roi_h = rt_h
                
                # Use current_frame if available (it might be old if captured at start of func?)
                # best to re-grab or reuse if scene static. Reusing `current_frame` from start.
                if current_frame is not None:
                     fh, fw = current_frame.shape[:2]
                     if roi_x + roi_w <= fw and roi_y + roi_h <= fh:
                         roi = current_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                         
                         # Preprocess for OCR
                         roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                         _, roi_thresh = cv2.threshold(roi_gray, 148, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                         
                         # Debug
                         # cv2.imwrite(os.path.join(base_path, "debug_ocr_roi.png"), roi_thresh)
                         
                         try:
                             # Tesseract
                             text = pytesseract.image_to_string(roi_thresh, config='--psm 7 digits')
                             print(f"OCR Raw Text: '{text.strip()}'")
                             
                             # Parse number - Take first 2 digits
                             import re
                             digits = re.findall(r'\d', text)
                             if len(digits) >= 2:
                                 val_str = "".join(digits[:2])
                                 val = int(val_str)
                                 
                                 # Validate
                                 if 25 <= val <= 75:
                                     self.prob_label.configure(text=f"Current Chance: {val}%")
                                     ocr_active = True
                                     
                                     # Color
                                     c = "#3b82f6"
                                     if val <= 35: c = "#ef4444"
                                     elif val <= 55: c = "#f59e0b"
                                     self.prob_label.configure(text_color=c)
                         except Exception as e:
                             print(f"OCR Error: {e}")

        else:
            msg = "Anchor Not Found"
            color = "red"
            
        self.status_btn.configure(text=msg, text_color=color)
        if not "Found" in msg:
             self.after(2000, lambda: self.status_btn.configure(text="Recognize and Start", text_color="#374151"))

        # Calculate Probability (Fallback if OCR didn't update it)
        if not ocr_active:
             print("OCR failed or not found, using estimation.")
             self.calculate_probability()
             
        # Run Optimization (Merged)
        self.run_optimization()

    def update_preview(self):
        """No-op for continuous preview. We use static preview now."""
        pass
        
    def _show_static_preview(self, frame):
        """Updates the preview label with the given frame."""
        import cv2
        from PIL import Image
        if frame is None: return
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        
        # Resize to fit 400x300
        w, h = im_pil.size
        ratio = min(400/w, 300/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        im_pil = im_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        ctk_img = ctk.CTkImage(light_image=im_pil, dark_image=im_pil, size=(new_w, new_h))
        self.preview_label.configure(text="", image=ctk_img)
