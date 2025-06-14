import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

class FaceDetectionApp:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def load_known_faces(self, faces_dir="known_faces"):
        """Load known faces from a directory"""
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print(f"Created directory: {faces_dir}")
            print("Please add known face images to this directory")
            return

        for filename in os.listdir(faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                # Get the person's name from the filename (without extension)
                name = os.path.splitext(filename)[0]
                
                # Load image file
                image_path = os.path.join(faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encoding
                face_encoding = face_recognition.face_encodings(image)
                
                if face_encoding:
                    self.known_face_encodings.append(face_encoding[0])
                    self.known_face_names.append(name)
                    print(f"Loaded face: {name}")

    def run(self):
        """Run the face detection and recognition application"""
        # Initialize video capture
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("Error: Could not open video capture device")
            return

        print("Starting face detection and recognition...")
        print("Press 'q' to quit")

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Resize frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert the image from BGR color to RGB color
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame to save time
            if self.process_this_frame:
                # Find all the faces and face encodings in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]

                    self.face_names.append(name)

            self.process_this_frame = not self.process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceDetectionApp()
    app.load_known_faces()
    app.run() 