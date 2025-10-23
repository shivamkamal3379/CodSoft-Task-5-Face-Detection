# Deep Learning Face Detection and Recognition Script
# This script demonstrates how to use the high-level 'deepface' library 
# to build a face recognition system using pre-trained models like ArcFace.

import cv2
import os
from deepface import DeepFace
import numpy as np

# --- 1. CONFIGURATION ---
# Define the path to your database folder.
# IMPORTANT: This folder must contain subfolders, where each subfolder name 
# is the identity (person's name) and contains their face images.
DB_PATH = "my_face_db"

# Define the path to the image you want to test recognition on.
TEST_IMAGE_PATH = os.path.join(DB_PATH, "Unknown_Face", "test_image.jpg")

# The model we use for generating face embeddings (ArcFace is state-of-the-art).
MODEL_NAME = "ArcFace" 

# The distance metric (Cosine similarity is standard for ArcFace/FaceNet).
DISTANCE_METRIC = "cosine"

# --- FIX: HARDCODED THRESHOLD ---
# The standard similarity threshold for ArcFace using Cosine distance.
# Distance below this value is considered a match.
ARC_FACE_THRESHOLD = 0.68

# --- 2. DATABASE INITIALIZATION ---
def initialize_database():
    """
    Initializes the database by calculating embeddings for all faces in DB_PATH.
    This step is automatically handled by DeepFace.find() on the first run 
    and saves the results to a file (representations_vgg_face.pkl) for fast access later.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database folder '{DB_PATH}' not found. Please create it and add face images.")
        return

    print("--- Initializing Face Database (may take a few minutes on first run) ---")
    
    # We call find() once with a dummy image just to ensure the DB is indexed.
    try:
        DeepFace.find(
            img_path=TEST_IMAGE_PATH,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False, # Allows DeepFace to work even if detection fails on dummy image
            silent=True # Suppress console output during initialization
        )
        print("Database successfully indexed or already exists!")
    except Exception as e:
        # NOTE: Deepface will automatically download models on first run, which is expected.
        # This error is okay if the index file is created.
        print(f"An error occurred during DB initialization: {e}")


# --- 3. RECOGNITION FUNCTION ---
def recognize_face_in_image(img_path):
    """
    Performs the full detection and recognition pipeline on a single image.
    """
    print(f"\n--- Running Recognition on: {img_path} ---")
    
    # Use the hardcoded threshold
    threshold = ARC_FACE_THRESHOLD
    
    try:
        # DeepFace.find is the single function that does all the work:
        # 1. Detects faces in img_path.
        # 2. Extracts ArcFace embeddings for those faces.
        # 3. Compares embeddings against the pre-calculated DB embeddings.
        result = DeepFace.find(
            img_path=img_path,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True # We require a face to be detected now
        )
        
        # The result is a list of Pandas DataFrames (one per detected face).
        # We need to explicitly check if the DataFrame is empty.
        if len(result) > 0 and not result[0].empty:
            # We focus on the first detected face for simplicity
            df = result[0]
            
            # Use the reliable 'distance' column
            distance = df['distance'].iloc[0]
            
            # The 'identity' column holds the path to the matching image in the database.
            # Example path: 'my_face_db/Elon_Musk/elon1.jpg'
            identity_path = df['identity'].iloc[0]
            
            # Extract the name (the subfolder name)
            person_name = os.path.basename(os.path.dirname(identity_path))
            
            # Display match status
            if distance < threshold:
                print(f"✅ Recognized Identity: {person_name}")
                print(f"   Similarity Distance ({DISTANCE_METRIC}, lower is better): {distance:.4f} (Threshold: {threshold:.4f})")
                display_result(img_path, person_name, distance, threshold)
            else:
                print(f"❌ Face Detected, but UNKNOWN IDENTITY.")
                print(f"   Closest match distance: {distance:.4f} (Exceeds Threshold: {threshold:.4f})")
                display_result(img_path, "UNKNOWN", distance, threshold)

        else:
            # This branch means a face was detected, but no identity matched the threshold.
            print("❌ No face detected in the image, or no match found in the database.")
            
    except Exception as e:
        # Catching the case where no face is detected at all (DeepFace raises exception)
        print(f"❌ Recognition Failed (Ensure your image path is correct and contains a clear face): {e}")


# --- 4. VISUALIZATION (Requires a working environment with OpenCV) ---
def display_result(img_path, name, distance, threshold):
    """Loads the image, draws a bounding box, and displays the name/distance."""
    try:
        img = cv2.imread(img_path)
        
        # We use the 'opencv' detector backend for consistent bounding box retrieval
        detected_faces = DeepFace.extract_faces(img_path=img_path, detector_backend='opencv')
        
        if detected_faces:
            face_info = detected_faces[0] # Assuming we track the first detected face
            x, y, w, h = face_info['facial_area']['x'], face_info['facial_area']['y'], face_info['facial_area']['w'], face_info['facial_area']['h']
            
            # Color and label logic
            if distance < threshold:
                color = (0, 255, 0)  # Green for Match
                label = f"{name} ({distance:.2f} Dist)"
            else:
                color = (0, 0, 255)  # Red for Unknown
                label = f"UNKNOWN ({distance:.2f} Dist)"
                
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Draw label
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Display the image (requires a local environment)
            window_name = f"Face Recognition Result for {os.path.basename(img_path)}"
            cv2.imshow(window_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Could not draw bounding box: No face detected by OpenCV detector.")

    except Exception as e:
        print(f"Visualization error (OpenCV window may not be supported or error in detection): {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Ensure the database is initialized
    initialize_database()
    
    # 2. Run the recognition test
    recognize_face_in_image(TEST_IMAGE_PATH)

    print("\nScript finished. If you encounter visualization errors, the recognition logic still ran successfully.")
