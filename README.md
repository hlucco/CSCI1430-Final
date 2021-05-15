## About

Final project submission for Brown University CSCI 1430 (Introduction to Computer Vision). Application allowing users to operate the cursor through movments of their head and eyes. The model being used for the facial feature recognition is found at `./shape_predictor_68_face_landmarks.dat`. OpenCV for camera input, DLib for facial feature recognition, pyautogui for cursor control interface.

## Usage

On startup there is a rectangle located at the top right corner of the frame and a red line connecting the center of the box to the user's nose. To calibrate, press ENTER. Once calibrated the box should be moved to the center of the user's face. You can now control the cursor movement with the movement of your head. To move the cursor, move your nose outside of the bounding box and the cursor will move to the point aligned with your nose. Left click is also supported. To left click, close both eyes for half a second.

## Get Started

`python -m venv .env`

`source .env/bin/activate`

`pip install -r requirements.txt`

`python main.py`

## Demo

- https://drive.google.com/file/d/1vD76FJbXkc2pfncb_ts3E3jArAK_iR_O/view?usp=sharing
