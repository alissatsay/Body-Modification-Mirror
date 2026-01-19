Warped Mirror Project — README

This project contains Python scripts that create a “warped mirror” effect using OpenCV and MediaPipe. The program detects your pose, warps the body around the hip area, segments you from the background, and composites you over a captured scene. Some versions of the script also save frames automatically.

Purpose of the main files:

*   The main script runs the webcam, applies the warp, performs segmentation, and displays the final composited output.
    
*   Helper functions inside the file compute the warp maps, extract pose landmarks, build the background, and blend everything together.
    
*   Some variations of the script add frame saving or additional logic for gesture-controlled background cutting.
    

Requirements:

*   Python 3.8 or newer.
    
*   Create the virtual environment: 
python3 -m venv .venv
source .venv/bin/activate

*   python -m pip install --upgrade pip setuptools wheel
*   Intsall requirements: pip install -r requirements.txt

    

How to run:

1.  Make sure your webcam is connected.
    
2.  Run the script from the terminal:python warped\_mirror.py
    
3.  When the program asks you to step out of the frame, move out so it can capture a clean background.
    
4.  Step back in front of the camera and the warped mirror will appear.
    

How to quit:

*   Press the “q” key while the window is active to close the program.