from imutils import face_utils
import numpy as np
import pyautogui as pag
import dlib
import cv2
import math

yellow = (0, 255, 255)
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
origin = (0, 0)


def run():
    shape_predictor = "./shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

    vid = cv2.VideoCapture(1)
    resolution_w = 1280
    resolution_h = 720
    cam_w = 640
    cam_h = 480
    unit_w = resolution_w / cam_w
    unit_h = resolution_h / cam_h
    frame_count = 0
    eye_closed_prev = False

    while True:
        global origin
        # Grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        _, frame = vid.read()
        frame = cv2.flip(frame, 1)
        # frame = imutils.resize(frame, width=cam_w, height=cam_h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        # Loop over the face detections
        if len(rects) > 0:
            rect = rects[0]
        else:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if (key == 27):
                break
            continue

        # Determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        nose = shape[nStart:nEnd]

        # Frame has been flipped so right and left are inverted
        temp = leftEye
        leftEye = rightEye
        rightEye = temp

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        diff_ear = np.abs(leftEAR - rightEAR)

        nose_point = (nose[3, 0], nose[3, 1])

        # Compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, yellow, 1)
        cv2.drawContours(frame, [rightEyeHull], -1, yellow, 1)

        cv2.putText(frame, "press ENTER to calibrate", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)

        key = cv2.waitKey(1)
        if(key == 13):
            origin = nose_point

        x, y = origin
        newx, newy = nose_point
        w, h = 50, 30
        multiple = 1
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), green, 2)
        cv2.line(frame, origin, nose_point, red, 2)

        dir = direction(nose_point, origin, w, h)
        cv2.putText(frame, dir.upper(), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
        drag = 15
        if dir == 'right':
            pag.moveRel(drag, 0)
        elif dir == 'left':
            pag.moveRel(-drag, 0)
        elif dir == 'up':
            pag.moveRel(0, -drag)
        elif dir == 'down':
            pag.moveRel(0, drag)

        if ear > 5:
            if eye_closed_prev == True:
                frame_count += 1
            # change the eye_closed in this boolean
            eye_closed_prev = True
            if frame_count > 6:
                print("click")
                pag.click(button='left')
                # reset the count
                frame_count = 0
        else:
            # if the eye wasn't closed, this boolean will be False so the
            # next iteration knows that
            eye_closed_prev = False

        for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
            cv2.circle(frame, (x, y), 2, yellow, -1)

        cv2.imshow("Frame", frame)
        if (key == 27):
            break


def direction(nose_point, anchor_point, w, h, multiple=1):
    nx, ny = nose_point
    x, y = anchor_point

    if nx > x + multiple * w:
        return 'right'
    elif nx < x - multiple * w:
        return 'left'

    if ny > y + multiple * h:
        return 'down'
    elif ny < y - multiple * h:
        return 'up'

    return '-'


def eye_aspect_ratio(eye):
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    left = (eye[0][0], eye[0][1])
    right = (eye[3][0], eye[3][1])
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    top = middle_point(eye[1], eye[2])
    bottom = middle_point(eye[5], eye[4])
    # getting the width and height of the eye to get the ratio from the points in the detector
    eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
    eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

    if (eye_height != 0):
        ratio = eye_width / eye_height
    else:
        ratio = None
    return ratio


def middle_point(p1, p2):
    x = int((p1[0] + p2[0]) / 2)
    y = int((p1[1] + p2[1]) / 2)
    return (x, y)


run()
