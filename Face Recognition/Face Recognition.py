# Importing the libraries
import cv2

# Loading the haar-cascade modules
face_cascade = cv2.CascadeClassifier('./src/haarcascade_frontalface_default.xml') #  for face
eye_cascade = cv2.CascadeClassifier('./src/haarcascade_eye.xml') # for eyes
smile_cascade = cv2.CascadeClassifier('./src/haarcascade_smile.xml') # for smile

def detect(gray, frame): 
    '''
    The cascade works on gray scale and the final output(face detection) should be in the original image
    Input: Gray scale image and original image(sinlge images coming from webcam one-by-one)
    Output: Image with detector reactangles

    '''
    #------------------------- Detecting the face -------------------------
    
    # applying the detectMultiScale method from the face cascade to locate one or several faces in the image
    faces = face_cascade.detectMultiScale(gray, 
                                          1.3, # scaling factor
                                          5 # minimum number of neighbors
                                         ) 
    for (x, y, w, h) in faces: # For each detected face
        # x and y: coordinates, 
        # w: width of rectangle, 
        # h: height of rectangle
        
        # creating rectanlge around the face
        cv2.rectangle(frame, 
                      (x, y), # coordinates of the upperleft corner of the rectangle
                      (x + w, y + h), # lower-right corner of the rectangle
                      (255, 0, 0), # RGB color-code for rectangle
                      2 # thickness of the edges of the rectangle
                     )
        
        #------------------- Detecting the eyes in reference to the face -------------------
        
        # determining 2 region of interest each for gray-scale image and color image
        roi_gray = gray[y:y + h, x:x + w] # for black and white image(on which the algorithm applies)
        roi_color = frame[y:y + h, x:x + w] # for colored image(on which the actual face detection appears)
        
        # applying the detectMultiScale method to locate one or several eyes in the gray-scale image
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 50) 
        
        # creating a rectangle around the eyes, but inside the referential of the face
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,
                          (ex, ey),
                          (ex + ew, ey + eh),
                          (0, 255, 0), 
                          2)
            
        # ------------------- Detecting the smaile face in reference to the face -------------------
        
        # applying the detectMultiScale method to locate one or several eyes in the gray-scale image
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) 
        
        # creating a rectangle around the smiles, but inside the referential of the face
        for (sx, sy, sw, sh) in smiles: # For each detected smile:
            cv2.rectangle(roi_color,
                          (sx, sy),
                          (sx + sw, sy + sh),
                          (0, 0, 255), 
                          2)
    return frame # image with the detector rectangles

# Turning the webcam on to capture a video
video_capture = cv2.VideoCapture(0) # 0 refers to embedded webcam in the machine. 1 for external webcam

while 1: # running until keystroke
    _, frame = video_capture.read() # the last frame of the webcame(color image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # transforming the image to gray-scale
    canvas = detect(gray, frame)
    # displaying the output image with eye and face detected
    cv2.imshow('Video', canvas)
    
    # Breaking the loop with a keystroke of 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# turning the webcam off
video_capture.release()
# destroying all the windows inside which the images were displayed
cv2.destroyAllWindows()