# Face Detection and Recognition Application

This is a Python application that detects and recognizes faces in images. It uses the `face_recognition` library for face detection and recognition, and OpenCV for image processing.

## Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Create a directory called `known_faces` in the same directory as the script
2. Add photos of people you want to recognize to the `known_faces` directory
   - Use clear, front-facing photos
   - Name the files with the person's name (e.g., `john.jpg`, `sarah.png`)
   - Supported formats: .jpg, .jpeg, .png

3. There are two ways to use the application:

   a. Process a single image:
   ```python
   from face_detection_image import FaceDetectionImage
   
   app = FaceDetectionImage()
   app.load_known_faces()
   app.process_image("path/to/input/image.jpg", "path/to/output/image.jpg")
   ```

   b. Process all images in a directory:
   ```python
   from face_detection_image import FaceDetectionImage
   
   app = FaceDetectionImage()
   app.load_known_faces()
   app.process_directory("path/to/input/directory", "path/to/output/directory")
   ```

4. The application will:
   - Load the known faces from the `known_faces` directory
   - Process the input image(s)
   - Draw boxes around detected faces
   - Label recognized faces with their names
   - Label unknown faces as "Unknown"
   - Save the processed image(s) to the specified output location

## Features

- Face detection and recognition in images
- Support for multiple known faces
- Process single images or entire directories
- Automatic creation of output directories
- Clear visual feedback with boxes and labels
- Support for common image formats (.jpg, .jpeg, .png)

## Notes

- Make sure you have good quality images for better face detection
- The application works best with clear, front-facing photos in the known_faces directory
- The first time you run the application, it will create a `known_faces` directory if it doesn't exist
- Output images will be saved with "_detected" added to the filename if no output path is specified 