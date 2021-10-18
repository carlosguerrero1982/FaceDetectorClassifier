import os
import cv2 as cv
import time

import mediapipe as mp



cap = cv.VideoCapture('elon.mp4')

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
faceDet= mp_face_detection.FaceDetection()

# Reading the video file until finished
while (cap.isOpened()):

    # Capture frame-by-frame

    ret, frame = cap.read()

    imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results= faceDet.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
           # print(detection.location_data.relative_bounding_box)
            mp_drawing.draw_detection(frame,detection)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            cv.rectangle(frame,bbox,(255,0,255),2)
            cv.putText(frame,f'FPS:{int(detection.score[0] * 100)} %',(bbox[0],bbox[1] - 20),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)


    # if video finished or no Video Input
    if not ret:
        break

    # Our operations on the frame come here
    gray = frame

    # resizing the frame size according to our need
    gray = cv.resize(gray, (500, 300))

    # font which we will be using to display FPS
    font = cv.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)

    # displaying the frame with fps
    cv.imshow('frame', gray)

    # press 'Q' if you want to exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv.destroyAllWindows()