# Option 1: Using Ultralytics YOLOv8 (Recommended - Easy to install)
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO
import numpy as np

class YOLOv8Detector:
    def __init__(self):
        print("Loading YOLOv8 model...")
        # This will automatically download the model if not present
        self.model = YOLO('yolov8n.pt')  # yolov8n.pt is the nano version (fastest)
        print("YOLOv8 model loaded successfully!")
    
    def run_detection(self):
        """Run real-time detection with YOLOv8"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return
        
        print("Starting YOLOv8 real-time detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 inference
            results = self.model(frame, verbose=False)
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Display the frame
            cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Option 2: Using OpenCV's built-in cascade classifiers (Works without downloads)
class CascadeDetector:
    def __init__(self):
        # Load pre-trained cascade classifiers (built into OpenCV)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        print("Cascade classifiers loaded successfully!")
    
    def detect_objects(self, frame):
        """Detect faces, eyes, and bodies"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Detect bodies
        bodies = self.body_cascade.detectMultiScale(gray, 1.1, 4)
        
        return faces, eyes, bodies
    
    def draw_detections(self, frame, faces, eyes, bodies):
        """Draw bounding boxes"""
        # Draw faces in green
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw eyes in blue
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Eye', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw bodies in red
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Body', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def run_detection(self):
        """Run cascade detection"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return
        
        print("Starting Cascade detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            faces, eyes, bodies = self.detect_objects(frame)
            
            # Draw detections
            frame = self.draw_detections(frame, faces, eyes, bodies)
            
            # Add detection count
            total_detections = len(faces) + len(eyes) + len(bodies)
            cv2.putText(frame, f'Detections: {total_detections}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Cascade Object Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Option 3: Using MediaPipe (Google's ML framework)
# pip install mediapipe opencv-python

try:
    import mediapipe as mp
    
    class MediaPipeDetector:
        def __init__(self):
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_face = mp.solutions.face_detection
            self.mp_draw = mp.solutions.drawing_utils
            
            # Initialize detectors
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.5)
            
            print("MediaPipe detectors loaded successfully!")
        
        def run_detection(self):
            """Run MediaPipe detection"""
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open webcam!")
                return
            
            print("Starting MediaPipe detection...")
            print("Press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process detections
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)
                face_results = self.face_detection.process(rgb_frame)
                
                # Draw pose landmarks
                if pose_results.pose_landmarks:
                    self.mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Draw hand landmarks
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Draw face detections
                if face_results.detections:
                    for detection in face_results.detections:
                        self.mp_draw.draw_detection(frame, detection)
                
                # Display frame
                cv2.imshow('MediaPipe Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()

except ImportError:
    print("MediaPipe not installed. Install with: pip install mediapipe")
    MediaPipeDetector = None

# Main execution
if __name__ == "__main__":
    print("=== Real-time Object Detection ===")
    print("1. YOLOv8 (Best overall - auto downloads model)")
    print("2. Cascade Classifiers (No downloads needed)")
    print("3. MediaPipe (Body/Hand/Face detection)")
    
    choice = input("Choose detector (1, 2, or 3): ")
    
    try:
        if choice == "1":
            detector = YOLOv8Detector()
            detector.run_detection()
        elif choice == "2":
            detector = CascadeDetector()
            detector.run_detection()
        elif choice == "3":
            if MediaPipeDetector:
                detector = MediaPipeDetector()
                detector.run_detection()
            else:
                print("MediaPipe not available. Install with: pip install mediapipe")
        else:
            print("Invalid choice!")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying Cascade Detector as fallback...")
        detector = CascadeDetector()
        detector.run_detection()

# Alternative simple installation script
print("\n=== Installation Instructions ===")
print("For YOLOv8 (Recommended):")
print("pip install ultralytics opencv-python")
print("\nFor MediaPipe:")
print("pip install mediapipe opencv-python")
print("\nFor basic detection (already included in OpenCV):")
print("pip install opencv-python")