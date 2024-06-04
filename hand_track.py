import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

frames = 0

#CONFIG
#########
working_rate = 3
# Create a named window and set its size
#cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('MediaPipe Hands', 1920, 1080)  # You can adjust the size as needed

def get_num_hand(results):
  output = 0
  r_hand_landmarks = results.multi_hand_landmarks
  if r_hand_landmarks:
    return len(r_hand_landmarks)
  return output

# is it a right or left hand?
def get_hand_label(results, hand_id):
  label = "not found"
  num_hand = get_num_hand(results)
  if((hand_id < num_hand) and (hand_id >= 0)):
    #as the image is fliped, the label have to be inverted too
    label = "Left" if results.multi_handedness[hand_id].classification[0].label == "Right" else "Right"
  return label

def get_multi_hand_label(results):
  labels = {}
  num_hand = get_num_hand(results)
  for i in range(num_hand):
    labels[i] = (get_hand_label(results,i))
  return labels

def get_hand_point_landmark(results, hand_id, point):
  coordinate = ("not_found", "not_found")

  num_hand = get_num_hand(results)
  if (((hand_id < num_hand) and (hand_id >= 0)) and ((point < 21) and (point >= 0))):
      landmark = results.multi_hand_landmarks[hand_id].landmark[point]
      cx = landmark.x
      cy = landmark.y
      coordinate = (cx,cy)
  return coordinate

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

def get_multi_hand_landmarks(results):
  hands_landmarks_dict = {}

  num_hand = get_num_hand(results)
  for i in range(num_hand):
    hands_landmarks_dict[i] = get_hand_landmarks(results, i)
  return hands_landmarks_dict

def working_frame():
  global frames
  work = 0
  if(frames == (working_rate-1)):
    frames = 0
  else:
    if (frames == 0):
      work = 1
    frames = frames + 1
  return work

#def check_point

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

    #PREDICTION CODE
    ###########################

    if working_frame():
      print(get_hand_point_landmark(results, 0, 4))

      #print(get_multi_hand_landmarks(results))

      ## Write marks
      #num_hand = get_num_hand(results)
      ##print(num_hand)
  #
      #hand_ids = {}
  #
      #hand_id = 0
      #hand_label = get_hand_label(results, hand_id)
      #print(hand_label)
  #
      #hand_landmarks = get_hand_landmarks(results, hand_id)
      ##print(hand_landmarks)

    #END PREDICTION CODE
    ###########################

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
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