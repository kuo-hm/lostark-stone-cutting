import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui import StoneCutterApp

def main():
    """
    Entry point for the Lost Ark Stone Cutter application.
    Initializes and runs the main GUI loop.
    """
    app = StoneCutterApp()
    app.mainloop()

if __name__ == "__main__":
    main()
