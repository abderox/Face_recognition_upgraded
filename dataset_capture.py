# @author abdelhadi mouzafir & FIROUD Reda
# Import OpenCV2 for image processing
import cv2
import np
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


face_id = input('enter your id')
# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize sample face image
count = 0

assure_path_exists("dataset1/")

# Start looping
while (True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))

    # Loops for each faces
    for (x, y, w, h) in faces:
        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset1/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        ## Now we gonna augement our data by rotating the image to detect faces that are not alligned
        anglee = 45
        center = (w / 2, h / 2)
        scale = 1
        for z in range(anglee):
            M = cv2.getRotationMatrix2D(center, z, scale)
            N = cv2.getRotationMatrix2D(center, -z, scale)

            rotated = cv2.warpAffine(gray[y:y + h, x:x + w], M, (w, h))
            antirotated = cv2.warpAffine(gray[y:y + h, x:x + w], N, (w, h))
            cv2.imwrite("dataset1/User." + str(face_id) + '.' + str(count) + str(z) + ".jpg", rotated)
            cv2.imwrite("dataset1/User." + str(face_id) + '.' + str(count) + ' anti '+str(z) + ".jpg", antirotated)

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count >= 50:
        print("Successfully Captured")
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
