import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

class FaceDetectionImage:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

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

    def process_image(self, input_image_path, output_image_path=None):
        """Process a single image and save the result"""
        if not os.path.exists(input_image_path):
            print(f"Error: Input image not found at {input_image_path}")
            return False

        # Load the image
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Error: Could not read image at {input_image_path}")
            return False

        # Convert the image from BGR color to RGB color
        rgb_image = image[:, :, ::-1]

        # Find all the faces and face encodings in the image
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append(name)

        # Draw the results on the image
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # If no output path is specified, create one based on the input path
        if output_image_path is None:
            filename, ext = os.path.splitext(input_image_path)
            output_image_path = f"{filename}_detected{ext}"

        # Save the resulting image
        cv2.imwrite(output_image_path, image)
        print(f"Processed image saved to: {output_image_path}")
        return True

    def process_directory(self, input_dir, output_dir=None):
        """Process all images in a directory"""
        if not os.path.exists(input_dir):
            print(f"Error: Input directory not found at {input_dir}")
            return

        if output_dir is None:
            output_dir = os.path.join(input_dir, "detected")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        processed_count = 0
        for filename in os.listdir(input_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"detected_{filename}")
                
                if self.process_image(input_path, output_path):
                    processed_count += 1

        print(f"Processed {processed_count} images. Results saved in: {output_dir}")

if __name__ == "__main__":
    app = FaceDetectionImage()
    app.load_known_faces()
    
    # Example usage:
    # Process a single image
    # app.process_image("path/to/input/image.jpg", "path/to/output/image.jpg")
    
    # Process all images in a directory
    # app.process_directory("path/to/input/directory", "path/to/output/directory") 