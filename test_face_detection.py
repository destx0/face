import face_recognition
import cv2
import numpy as np
import os

def load_known_faces(known_faces_dir="known_faces"):
    """Load known faces from the known_faces directory"""
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print(f"Created {known_faces_dir} directory. Please add known face images there.")
        return known_face_encodings, known_face_names

    for filename in os.listdir(known_faces_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # Get the person's name from the filename (without extension)
            name = os.path.splitext(filename)[0]
            
            # Load image file
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)
                print(f"Loaded face: {name}")
            else:
                print(f"No face found in {filename}")

    return known_face_encodings, known_face_names

def test_face_detection_and_recognition():
    # Load known faces
    print("Loading known faces...")
    known_face_encodings, known_face_names = load_known_faces()
    
    if not known_face_encodings:
        print("No known faces loaded. Please add some face images to the 'known_faces' directory.")
        return

    # Create test directories if they don't exist
    if not os.path.exists("test_images"):
        os.makedirs("test_images")
        print("Created test_images directory. Please add some test images there.")
        return

    # Process each image in the test_images directory
    for filename in os.listdir("test_images"):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            print(f"\nProcessing {filename}...")
            
            # Load the image
            image_path = os.path.join("test_images", filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error: Could not read image {filename}")
                continue
                
            # Convert the image from BGR color to RGB color
            rgb_image = image[:, :, ::-1]
            
            # Find all the faces in the image
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                print(f"No faces found in {filename}")
                continue
                
            print(f"Found {len(face_locations)} face(s) in {filename}")
            
            # Get face encodings for the faces in the image
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            # Initialize list for face names
            face_names = []
            
            # Loop through each face in this image
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)
            
            # Draw boxes around the faces and label them
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw a box around the face
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            # Save the output image
            output_filename = f"detected_{filename}"
            output_path = os.path.join("test_images", output_filename)
            cv2.imwrite(output_path, image)
            print(f"Saved result to {output_filename}")

if __name__ == "__main__":
    test_face_detection_and_recognition() 