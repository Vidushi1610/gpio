import cv2
import dlib
import RPi.GPIO as GPIO
import time

# Set up GPIO
GPIO.setmode(GPIO.BCM)
drowsy_pin = 18  # Example GPIO pin, change to your desired pin
GPIO.setup(drowsy_pin, GPIO.OUT)

# Load the face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Load the facial landmarks predictor model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the Yawn and Blink detection models
yawn_blink_detector = dlib.simple_object_detector("yawn_blink_detector.svm")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    yawn_blink_detected = False

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_list = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

        # Detect yawning and blinking
        yawn_blink_rects = yawn_blink_detector(frame)

        for rect in yawn_blink_rects:
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

            # Capture image when yawning or blinking
            yawn_blink_image = frame[y:y+h, x:x+w]
            cv2.imwrite("yawn_blink_image.jpg", yawn_blink_image)

            yawn_blink_detected = True

            # Draw rectangle around detected yawn/blink
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Yawn and Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if yawn_blink_detected:
        print("Yawn or Blink detected. Image captured.")
        GPIO.output(drowsy_pin, GPIO.HIGH)
        time.sleep(1)  # GPIO pin HIGH for 1 second
        GPIO.output(drowsy_pin, GPIO.LOW)
        yawn_blink_detected = False

# Release GPIO and close all windows
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()
