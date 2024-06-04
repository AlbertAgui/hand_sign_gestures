import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Create a named window and set its size
#cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('MediaPipe Hands', 1920, 1080)  # You can adjust the size as needed

def get_num_hand(results):
  output = 0
  r_hand_landmarks = results.multi_hand_landmarks
  if r_hand_landmarks:
    return len(r_hand_landmarks)
  return output

def get_hand_label(results, hand_id):
  label = "not found"
  num_hand = get_num_hand(results)
  if((hand_id < num_hand) and (hand_id >= 0)):
    label = results.multi_handedness[hand_id].classification[0].label
  return label

#return dict accessed by landmark id
def get_hand_landmarks(results, hand_id):
  hand_landmarks_dict = {}

  num_hand = get_num_hand(results)
  if((hand_id < num_hand) and (hand_id >= 0)):
    for ids, landmark in enumerate(results.multi_hand_landmarks[hand_id].landmark):
      cx = landmark.x
      cy = landmark.y
      hand_landmarks_dict[ids] = (cx, cy)

  return hand_landmarks_dict

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

    # contains right or left side + prediction score
    # only scope >0.5 are valid points
    r_handedness     = results.multi_handedness
    # contains coordinates (all points are always valid if only one of them is valid
    #                      so those have to be treaten)
    r_hand_landmarks = results.multi_hand_landmarks

    # Use hand points and predictions
    # if there are results
    # the ids are respect to which hand we have
    #if r_handedness:
    #  for ids, handedness in enumerate(r_handedness):
    #    label = handedness.classification[0].label
    #    #score = handedness.classification[0].score
    #    index = handedness.classification[0].index
    #    print (index, label)

    # Write marks
    num_hand = get_num_hand(results)
    #print(num_hand)

    hand_id = 0
    hand_label = get_hand_label(results, hand_id)
    #print(hand_label)

    hand_landmarks = get_hand_landmarks(results, hand_id)
    print(hand_landmarks)

    #  for i in range(len(r_hand_landmarks)):
    #    # Index
    #    # Here is How to Get All the Coordinates
    #    for ids, landmrk in enumerate(i.landmark):
    #        cx, cy = landmrk.x * image_width, landmrk.y*image_height
    #        hand_id = i
    #        #hand_side = results.multi_handedness[ids].classification[0].label
    #      #  print (ids, cx, cy)

    if r_hand_landmarks:
      for hand_landmarks in r_hand_landmarks:
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