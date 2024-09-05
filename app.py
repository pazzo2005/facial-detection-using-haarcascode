from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Load face and eye classifiers once (before starting the loop)
detector = cv2.CascadeClassifier(r'E:\webcam detection\haarcascade\haarcascade_frontalcatface_extended.xml')
eye_cascade = cv2.CascadeClassifier(r'E:\webcam detection\haarcascade\haarcascade_eye (1).xml')

# Check if the cascades are loaded properly
if detector.empty():
    print("Error loading face detector cascade.")
    exit()  # Exit the program if the cascade is not loaded properly

if eye_cascade.empty():
    print("Error loading eye detector cascade.")
    exit()  # Exit the program if the cascade is not loaded properly

def generate_frames():
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()

        if not success:
            continue  # Skip the frame if it fails to capture

        # Convert the frame to grayscale for better face and eye detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle for face

            # Region of interest for detecting eyes within the face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)  # Blue rectangle for eyes

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
