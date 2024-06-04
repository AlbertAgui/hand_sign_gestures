import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Create a named window and set its size
#cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('MediaPipe Hands', 1920, 1080)  # You can adjust the size as needed

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    #get image size
    image_height, image_width, _ = image.shape

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # it just checks if there are hand marks
    if results.multi_hand_landmarks:
      
      for hand_landmarks in results.multi_hand_landmarks:
        # Index
        # Here is How to Get All the Coordinates
        for ids, landmrk in enumerate(hand_landmarks.landmark):
            cx, cy = landmrk.x * image_width, landmrk.y*image_height
            #hand_side = results.multi_handedness[ids].classification[0].label
            print (ids, cx, cy)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()

# Destroy all the windows 
cv2.destroyAllWindows() 