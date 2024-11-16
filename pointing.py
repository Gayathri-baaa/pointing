import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to detect the gesture
def recognize_gesture(landmarks):
    # Get the position of the landmarks of the hand (e.g., the tips of the fingers)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Define gesture recognition logic (example for open hand and fist)
    # Simple gesture classification logic based on landmarks' positions
    
    # Example: Gesture 1 - Open Hand (fingers spread out)
    if (index_tip[1] < thumb_tip[1] and middle_tip[1] < index_tip[1] and
        ring_tip[1] < middle_tip[1] and pinky_tip[1] < ring_tip[1]):
        return "Open Hand"
    
    # Example: Gesture 2 - Fist (fingers curled in)
    if (index_tip[1] > thumb_tip[1] and middle_tip[1] > index_tip[1] and
        ring_tip[1] > middle_tip[1] and pinky_tip[1] > ring_tip[1]):
        return "Fist"
    
    # Example: Gesture 3 - Pointing (index finger extended)
    if (index_tip[1] < thumb_tip[1] and middle_tip[1] > index_tip[1] and
        ring_tip[1] > middle_tip[1] and pinky_tip[1] > ring_tip[1]):
        return "Pointing"

    return "Unknown Gesture"


# OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the image horizontally for better display
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # If hand landmarks are found
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmark coordinates (x, y) relative to the image size
            hand_landmarks = [(lm.x, lm.y) for lm in landmarks.landmark]

            # Recognize gesture based on the landmarks
            gesture = recognize_gesture(hand_landmarks)
            
            # Display the gesture text on the frame
            cv2.putText(frame, f'Gesture: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the output frame
    cv2.imshow("Hand Gesture Recognition", frame)
    
    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
