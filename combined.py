import cv2
import dlib
import RPi.GPIO as GPIO
from imutils import face_utils

# GPIO setup
GPIO_PIN = 17  # Change this to the appropriate GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_PIN, GPIO.OUT)

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

# Yawning thresholds
EAR_THRESHOLD = 0.2  # Eye aspect ratio threshold
YAWN_CONSEC_FRAMES = 15  # Number of consecutive frames for yawn detection

# Calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate Euclidean distance between two points
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Start capturing video
cap = cv2.VideoCapture(0)

frame_count = 0
yawn_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[42:48]
        right_eye = shape[36:42]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            yawn_frames += 1
            if yawn_frames >= YAWN_CONSEC_FRAMES:
                GPIO.output(GPIO_PIN, GPIO.HIGH)  # Turn on alert
        else:
            yawn_frames = 0
            GPIO.output(GPIO_PIN, GPIO.LOW)  # Turn off alert

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
