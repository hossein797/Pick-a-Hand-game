import numpy as np
import cv2
import time
import random
import imutils
from cvzone.HandTrackingModule import HandDetector

# Initialize hand detector
detector = HandDetector(detectionCon=0.5, maxHands=2)

# Initialize variables
startGame = False
scores = [0, 0]
initial_time = None
chosen_hand = None

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to detect hand
def detect_hand(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if len(contours) > 0:
        # Find the largest contour (the hand)
        hand_contour = max(contours, key=cv2.contourArea)
        return hand_contour
    else:
        return None

# Function to detect ball in hand
def detect_ball_in_hand(hand_region):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    yellowLower, yellowUpper = (20, 100, 100), (30, 255, 255)
    mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),
            (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            return True
        else:
            return False
        
        
       
# Function to start the game
def start_game():
    global startGame, initial_time
    startGame = True
    initial_time = time.time()

# Function to reset game variables
def reset_game():
    global startGame, initial_time, chosen_hand
    startGame = False
    initial_time = None
    chosen_hand = None

# Main loop
while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hands in the frame
    hands, _ = detector.findHands(frame)

    # If two hands are detected, start the game
    if len(hands) == 2 and not startGame:
        start_game()

    # If the game is ongoing
    if startGame:
        # Calculate timer
        timer = int(5 - (time.time() - initial_time))

        # If timer reaches 0, choose a random hand and wait for the user to open it
        if timer > 0 and chosen_hand is None:
            chosen_hand = random.choice(["Left", "Right"])
            print(f"Chosen Hand: {chosen_hand}")
        
        # Detect open hand in the chosen hand
        # Inside the main loop, update the hand detection logic and open/closed hand detection

        # Detect open hand in the chosen hand
        # Inside the main loop, update the open/closed hand detection logic

    # Detect open hand in the chosen hand
    if chosen_hand and timer < 0:
        for hand in hands:
            # Get hand landmarks
            lmList = hand["lmList"]
            if lmList:
                # Get the tip of the index finger (landmark 8) and the base of the hand (wrist) (landmark 0)
                tip_index = lmList[8]
                wrist = lmList[0]
                
                # Calculate the Euclidean distance between the tip of the index finger and the base of the hand
                distance = np.linalg.norm(np.array(tip_index) - np.array(wrist))
                
                # Define a threshold to determine whether the hand is open or closed
                threshold = 100  # Adjust this threshold as needed
                
                # Determine if the hand is open or closed based on the distance
                if distance > threshold:
                    text = "Open"
                else:
                    text = "Closed"
                
                # Draw the text dynamically at the top of the bounding box
                cv2.putText(frame, text, (wrist[0], wrist[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0) if text == "Open" else (0, 0, 255), 2)

                # If the chosen hand is open, check if a ball is detected in it
                if text == "Open" and hand["type"] == chosen_hand:
                    ball_detected = detect_ball_in_hand(chosen_hand)
                    if ball_detected:
                        scores[0] += 1
                        print("Ball detected in chosen hand. Score for User+1.")
                    else:
                        scores[1] += 1
                        print("Ball not detected in chosen hand. Score for Computer+1.")
                    reset_game()


        
    # Display scores and timer on the frame
    cv2.putText(frame, f"Player: {scores[0]}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Computer: {scores[1]}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Chosen Hand: {chosen_hand}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Timer: {timer}" if startGame else "Waiting for Hands", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()