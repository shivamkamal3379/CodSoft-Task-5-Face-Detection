

import cv2
import os
from deepface import DeepFace
import numpy as np

# --- 1. CONFIGURATION ---

DB_PATH = "my_face_db"

 TEST_IMAGE_PATH = os.path.join(DB_PATH, "Unknown_Face", "test_image.jpg")

 MODEL_NAME = "ArcFace" 

 DISTANCE_METRIC = "cosine"

 
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
    
     try:
        DeepFace.find(
            img_path=TEST_IMAGE_PATH,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False, 
            silent=True 
        )
        print("Database successfully indexed or already exists!")
    except Exception as e:
        
        print(f"An error occurred during DB initialization: {e}")


def recognize_face_in_image(img_path):
    """
    Performs the full detection and recognition pipeline on a single image.
    """
    print(f"\n--- Running Recognition on: {img_path} ---")
    
    threshold = ARC_FACE_THRESHOLD
    
    try:
        ngs.
        result = DeepFace.find(
            img_path=img_path,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True # We require a face to be detected now
        )
        
        
        if len(result) > 0 and not result[0].empty:
            
            df = result[0]
            
            
            distance = df['distance'].iloc[0]
            
            
            identity_path = df['identity'].iloc[0]
            
            
            person_name = os.path.basename(os.path.dirname(identity_path))
            
            
            if distance < threshold:
                print(f"✅ Recognized Identity: {person_name}")
                print(f"   Similarity Distance ({DISTANCE_METRIC}, lower is better): {distance:.4f} (Threshold: {threshold:.4f})")
                display_result(img_path, person_name, distance, threshold)
            else:
                print(f"❌ Face Detected, but UNKNOWN IDENTITY.")
                print(f"   Closest match distance: {distance:.4f} (Exceeds Threshold: {threshold:.4f})")
                display_result(img_path, "UNKNOWN", distance, threshold)

        else:
           
            print("❌ No face detected in the image, or no match found in the database.")
            
    except Exception as e:
        
        print(f"❌ Recognition Failed (Ensure your image path is correct and contains a clear face): {e}")



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
