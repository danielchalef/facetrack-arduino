#!/usr/bin/python
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them,
then centers the webcam via two servos so the face is at the center of the screen
Based on facedetect.py in the OpenCV samples directory
"""
import sys, math, time
import cv2
import numpy as np
import Arduino.arduino as a

cv = cv2.cv

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned
# for accurate yet slow object detection. For a faster operation on real video
# images the settings are:
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING,
# min_size=<minimum possible face size
 
min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = cv.CV_HAAR_DO_CANNY_PRUNING

max_pwm = 165
min_pwm = 20
midScreenWindow = 30  # acceptable 'error' for the center of the screen.
panStepSize = 2 # degree of change for each pan update
tiltStepSize = 2 # degree of change for each tilt update
servoPanPosition = 90 # initial pan position
servoTiltPosition = 30 # initial tilt position

TILT_PIN = 9
PAN_PIN = 10

board = a.Arduino('115200', port="/dev/ttyACM0")
servo = a.Servos(board)

servo.attach(TILT_PIN)
servo.attach(PAN_PIN)

     
def detect_and_draw(img, cascade):
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
                   cv.Round (img.height / image_scale)), 8, 1)
 
    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
 
    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)
 
    cv.EqualizeHist(small_img, small_img)
 
    midFace = None
 
    if(cascade):
        t = cv.GetTickCount()
        # HaarDetectObjects takes 0.02s
        faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, haar_flags, min_size)
        t = cv.GetTickCount() - t
        if faces:
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the
                # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
                # get the xy corner co-ords, calc the midFace location
                x1 = pt1[0]
                x2 = pt2[0]
                y1 = pt1[1]
                y2 = pt2[1]
                midFaceX = x1+((x2-x1)/2)
                midFaceY = y1+((y2-y1)/2)
                midFace = (midFaceX, midFaceY)
 
    cv.ShowImage("result", img)
    return midFace
    
def move(SERVO_PIN, angle):
    if (min_pwm <= angle <= max_pwm):
        servo.write(SERVO_PIN, angle)
        time.sleep(0.2) # give the servo time to get to destination
    else:
        print "Servo angle must be an integer between min and max pwm.\n"
    

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def np_array_to_iplimage(source):
    # convert numpy image array back to iplimage 
    bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 3)
    cv.SetData(bitmap, source.tostring(), 
               source.dtype.itemsize * 3 * source.shape[1])
    return(bitmap)

if __name__ == '__main__':

    HAAR_CASCADE = "haarcascade_frontalface_alt.xml"

    cascade = cv.Load(HAAR_CASCADE)
    capture = cv.CreateCameraCapture(0)

    cv.NamedWindow("result", 1)
 
    move(PAN_PIN, servoPanPosition)
    move(TILT_PIN, servoTiltPosition)

    if capture:
        frame_copy = None
 
        while True:
            frame = cv.QueryFrame(capture)
            # My webcam is sideways. Rorate -90 degrees
            frame = np_array_to_iplimage(rotate_about_center(np.asarray(frame[:,:]), -90))
            
            if not frame:
                cv.WaitKey(0)
                break
            if not frame_copy:
                frame_copy = cv.CreateImage((frame.width,frame.height),
                                            cv.IPL_DEPTH_8U, frame.nChannels)
            if frame.origin == cv.IPL_ORIGIN_TL:
                cv.Copy(frame, frame_copy)
            else:
                cv.Flip(frame, frame_copy, 0)
            
            # where is the middle of the screen?
            midScreenX = (frame.width/2)
            midScreenY = (frame.height/2)
  
            # try find a face, return the midface coordinates
            midFace = detect_and_draw(frame_copy, cascade)

            if midFace is not None:
                LAST_DETECTED = True
                midFaceX = midFace[0]
                midFaceY = midFace[1]
                
                #Find out if the X component of the face is to the left of the middle of the screen.
                if(midFaceX < (midScreenX - midScreenWindow)):
                    #Update the pan position variable to move the servo to the right.
                    servoPanPosition += panStepSize
                    print str(midFaceX) + " > " + str(midScreenX) + " : Pan Right : " + str(servoPanPosition)
                #Find out if the X component of the face is to the right of the middle of the screen.
                elif(midFaceX > (midScreenX + midScreenWindow)):
                    #Update the pan position variable to move the servo to the left.
                    servoPanPosition -= panStepSize
                    print str(midFaceX) + " < " + str(midScreenX) + " : Pan Left : " + str(servoPanPosition)
                else:
                    print str(midFaceX) + " ~ " + str(midScreenX) + " : " + str(servoPanPosition)
                
                servoPanPosition = min(servoPanPosition, max_pwm)
                servoPanPosition = max(servoPanPosition, min_pwm)               
                move(PAN_PIN, servoPanPosition)
                
                #Find out if the Y component of the face is below the middle of the screen.
                if(midFaceY < (midScreenY - midScreenWindow)):
                    if(servoTiltPosition <= max_pwm):
                        #Update the tilt position variable to lower the tilt servo.
                        servoTiltPosition -= tiltStepSize
                        print str(midFaceY) + " > " + str(midScreenY) + " : Tilt Down : " + str(servoTiltPosition)
                #Find out if the Y component of the face is above the middle of the screen.
                elif(midFaceY > (midScreenY + midScreenWindow)):
                    if(servoTiltPosition >= 1):
                        #Update the tilt position variable to raise the tilt servo.
                        servoTiltPosition += tiltStepSize
                        print str(midFaceY) + " < " + str(midScreenY) + " : Tilt Up : " + str(servoTiltPosition)
                else:
                    print str(midFaceY) + " ~ " + str(midScreenY) + " : " + str(servoTiltPosition)
                
                servoTiltPosition = min(servoTiltPosition, max_pwm)
                servoTiltPosition = max(servoTiltPosition, min_pwm)  
                move(TILT_PIN, servoTiltPosition)
                
            if cv.WaitKey(10) >= 0: # 10ms delay
                break
 
    cv.DestroyWindow("result")

    servo.detach(TILT_PIN)
    servo.detach(PAN_PIN)
    board.close()
