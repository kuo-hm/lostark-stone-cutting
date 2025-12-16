import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load env
load_dotenv()

class GeminiAdvisor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        else:
            self.model = None

    def get_suggestion(self, state, goal_mode, target_line):
        """
        Queries Gemini for the best move based on the provided state dict.
        Returns: (best_move, reason)
        """
        if not self.model:
            return None, "Error: No GEMINI_API_KEY found"

        prompt = f"""
        I am cutting a Lost Ark Ability Stone.
        Current State: {state}
        
        Game Rules:
        - Base Success Chance starts at 75%.
        - Success decreases chance by 10%.
        - Fail increases chance by 10%.
        - Caps: Min 25%, Max 75%.
        
        My Goal:
        - Mode: {goal_mode} (e.g., 77 means 7/7, 97 means 9/7).
        - Target Line for 9 (if 97 mode): {target_line}.
        
        Task:
        Suggest the optimally best NEXT single click.
        Options: "Line 1", "Line 2", "Malice".
        
        Output format strictly as JSON:
        {{
            "move": "Line 1",
            "reason": "Brief explanation why."
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            txt = response.text
            
            import json
            # Handle potential markdown fencing
            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0]
            elif "```" in txt:
                txt = txt.split("```")[1].split("```")[0]
                
            data = json.loads(txt)
            return data.get("move", "Line 1"), data.get("reason", "AI Suggestion")
            
        except Exception as e:
            return None, f"API Error: {e}"
