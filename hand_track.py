import cv2
import mediapipe as mp
from tqdm import tqdm
import array
from time import sleep
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

#This frames are used for workig frames
"""w_frames = 0
working_rate = 20"""

#Tracking
#This frames are used for tracking hands progress
t_frames = 0
#contain the number of frames needed to track a hand
max_track_frames = 50
#contain the current tracked letters
MAX_LETTERS = 8
letters        = ""
current_letter = ""

# Font settings
font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 1.5
font_thickness = 3
font_color = (255, 255, 255)  # White color

# Test hello
"""lock1 = lock2 = lock3 = lock4 = lock5 = 0"""

# Create a named window and set its size
cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Hands', 1920, 1080)  # You can adjust the size as needed

# get number of hands found
def get_num_hand(results):
  output = 0
  r_hand_landmarks = results.multi_hand_landmarks
  #if result landmarks valid
  if r_hand_landmarks:
    return len(r_hand_landmarks)
  return output

# get hand label(right or left) for a certain hand_id
def get_hand_label(results, hand_id):
  label = "not found"
  num_hand = get_num_hand(results)
  #check if the hand_id exists
  if((hand_id < num_hand) and (hand_id >= 0)):
    #as the image is fliped, the label have to be inverted too
    label = "Left" if results.multi_handedness[hand_id].classification[0].label == "Right" else "Right"
  return label

# get hand label(right or left) of all hands
def get_multi_hand_label(results):
  labels = {}
  num_hand = get_num_hand(results)
  for i in range(num_hand):
    labels[i] = (get_hand_label(results,i))
  return labels

# get hand point coordinates
def get_hand_point_landmark(results, hand_id, point):
  coordinate = ("not_found", "not_found")

  num_hand = get_num_hand(results)
  #check if the hand_id and point are valid
  if (((hand_id < num_hand) and (hand_id >= 0)) and ((point < 21) and (point >= 0))):
    landmark = results.multi_hand_landmarks[hand_id].landmark[point]
    cx = landmark.x
    cy = landmark.y
    coordinate = (cx,cy)
  return coordinate

# get all hand point coordinates of hand_id hand
# return dict accessed by point id
def get_hand_landmarks(results, hand_id):
  hand_landmarks_dict = {}

  num_hand = get_num_hand(results)
  #check hand_id range
  if((hand_id < num_hand) and (hand_id >= 0)):
    for ids, landmark in enumerate(results.multi_hand_landmarks[hand_id].landmark):
      cx = landmark.x
      cy = landmark.y
      hand_landmarks_dict[ids] = (cx, cy)
  return hand_landmarks_dict

# get all hand point coordinates of all hands
def get_multi_hand_landmarks(results):
  hands_landmarks_dict = {}

  num_hand = get_num_hand(results)
  for i in range(num_hand):
    hands_landmarks_dict[i] = get_hand_landmarks(results, i)
  return hands_landmarks_dict

#This method uses normalized coordinates
#coordinate[0] is like coordinate.x
#coordinate[1] is like coordinate.y
def is_correct_point(results, hand_id, point):
  is_correct = 0
  coordinate = get_hand_point_landmark(results, hand_id, point)

  if ((coordinate != ("not_found", "not_found")) and 
     ((coordinate[0] >= 0 and coordinate[0] <= 1) and
      (coordinate[1] >= 0 and coordinate[1] <= 1))):
    is_correct = 1
  return is_correct

#This method checks the positon of point1 with respect to point2
#coordinate[0] is like coordinate.x
#coordinate[1] is like coordinate.y
#Operations supported:
#"Left"
#"Right"
#"Top"
#"Down"
#it is used to examine the position of the point 1 relative to point 0 of the hand for both x and y
def is_point_at(results, hand_id, point1, point2, operation):
  is_at = 0
  if (is_correct_point(results, hand_id, point1) and 
      is_correct_point(results, hand_id, point2)):
    coordinate1 = get_hand_point_landmark(results, hand_id, point1)
    coordinate2 = get_hand_point_landmark(results, hand_id, point2)
    if operation == "Left":
      if (coordinate1[0] > coordinate2[0]):
        is_at = 1
    elif operation == "Right":
      if (coordinate1[0] > coordinate2[0]):
        is_at = 1
    if operation == "Top":
      if (coordinate1[1] < coordinate2[1]):
        is_at = 1
    elif operation == "Down":
      if (coordinate1[1] > coordinate2[1]):
        is_at = 1
  return is_at
    
"""
def working_frame():
  global w_frames
  work = 0
  if(w_frames == (working_rate-1)):
    w_frames = 0
  else:
    if (w_frames == 0):
      work = 1
    w_frames = w_frames + 1
  return work

def detect_hello(label,lock1,lock2,lock3,lock4,lock5,progress_bar):
            if lock1 == 0:
                L1 = letter_H(results, hand_id)
                if L1 == 1:
                    lock1 = 1
                    progress_bar.update(1)

            if lock1 == 1 and lock2 == 0:
                L2 = letter_E(results, hand_id)
                if L2 == 1:
                    lock2 = 1
                    progress_bar.update(1)

            if lock2 == 1 and lock3 == 0:
                L3 = letter_L(results, hand_id)
                if L3 == 1:
                    lock3 = 1
                    progress_bar.update(1)
                    lock2 = 0  # Reset lock2 after updating the progress bar

            if lock3 == 1 and lock4 == 0:
                L4 = letter_L(results, hand_id)
                if L4 == 1:
                    lock4 = 1
                    progress_bar.update(1)
                    lock3 = 0  # Reset lock3 after updating the progress bar

            if lock4 == 1 and lock5 == 0:
                L5 = letter_O(results, hand_id)
                if L5 == 1:
                    lock5 = 1
                    progress_bar.update(1)
                    lock4 = 0  # Reset lock4 after updating the progress bar

            if lock5 == 1:
                print("Hello word complete!")
                progress_bar.n = 0
                progress_bar.last_print_n = 0
                progress_bar.refresh()
                lock1 = lock2 = lock3 = lock4 = lock5 = 0
                sleep(4)

def progress_bar(any_array):
  lock = array.array('i', [0] * len(any_array))
  with Bar('HELLO',max=5) as bar:
    for i in range(len(any_array)):
      if any_array[i] == 1:
        lock[i]=1
        if lock[i-1] == 1:
         bar.next()
    if lock[len(any_array)-1] == 1:
      print("HELLO WORD WAS DETECTED!!!!")
      lock = array.array('i', [0] * len(any_array))
"""

def letter_A(results,hand_id):
  A_letter=0
  if(is_point_at(results,hand_id,7,8,"Top") and is_point_at(results,hand_id,11,12,"Top")
  and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,19,20,"Top")
  and is_point_at(results,hand_id,4,8,"Top")and is_point_at(results,hand_id,4,3,"Top")
  and is_point_at(results,hand_id,2,1,"Top")):
   if(is_point_at(results,hand_id,4,8,"Left") and is_point_at(results,hand_id,3,7,"Left")
   and is_point_at(results,hand_id,2,6,"Left")and is_point_at(results,hand_id,2,5,"Left")):
     A_letter=1
     print("The letter represented is A") 
  return A_letter

def letter_B(results,hand_id):
  B_letter=0
  if(is_point_at(results,hand_id,8,7,"Top") and is_point_at(results,hand_id,12,11,"Top")
  and is_point_at(results,hand_id,16,15,"Top")and is_point_at(results,hand_id,20,19,"Top")
  and is_point_at(results,hand_id,12,16,"Top")and is_point_at(results,hand_id,16,8,"Top")
  and is_point_at(results,hand_id,8,20,"Top")and is_point_at(results,hand_id,7,10,"Top")
  and is_point_at(results,hand_id,7,14,"Top")and is_point_at(results,hand_id,7,18,"Top")):
   if(is_point_at(results,hand_id,3,9,"Left") and is_point_at(results,hand_id,1,4,"Left")):
   
     B_letter=1
     print("The letter represented is B") 
  return B_letter 


def letter_C(results,hand_id):
  C_letter=0
  if(is_point_at(results,hand_id,12,8,"Top") and is_point_at(results,hand_id,11,7,"Top")
  and is_point_at(results,hand_id,5,4,"Top")and is_point_at(results,hand_id,4,3,"Top")):
   if(is_point_at(results,hand_id,1,2,"Left") and is_point_at(results,hand_id,2,3,"Left")
   and is_point_at(results,hand_id,7,8,"Left")and is_point_at(results,hand_id,0,1,"Left")
   and is_point_at(results,hand_id,6,7,"Left")and is_point_at(results,hand_id,5,6,"Left")
   and is_point_at(results,hand_id,10,11,"Left")and is_point_at(results,hand_id,0,5,"Left")
   and is_point_at(results,hand_id,0,9,"Left")):
     C_letter=1
     print("The letter represented is C") 
  return C_letter 
  
def letter_D(results,hand_id):
  D_letter=0
  if(is_point_at(results,hand_id,11,12,"Top") and is_point_at(results,hand_id,15,16,"Top")
  and is_point_at(results,hand_id,19,20,"Top")and is_point_at(results,hand_id,8,7,"Top")
  and is_point_at(results,hand_id,7,6,"Top")and is_point_at(results,hand_id,6,4,"Top")
  and is_point_at(results,hand_id,11,4,"Top")and is_point_at(results,hand_id,15,4,"Top")
  and is_point_at(results,hand_id,3,2,"Top")and is_point_at(results,hand_id,2,1,"Top")
  and is_point_at(results,hand_id,1,0,"Top")and is_point_at(results,hand_id,5,4,"Top")):
   if(is_point_at(results,hand_id,3,4,"Left") and is_point_at(results,hand_id,5,9,"Left")
   and is_point_at(results,hand_id,9,13,"Left")and is_point_at(results,hand_id,13,17,"Left")):
     D_letter=1
     print("The letter represented is D") 
  return D_letter


def letter_E(results,hand_id):
  E_letter=0
  if(is_point_at(results,hand_id,7,8,"Top") and is_point_at(results,hand_id,11,12,"Top")
  and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,19,20,"Top")
  and is_point_at(results,hand_id,8,4,"Top")and is_point_at(results,hand_id,12,4,"Top")
  and is_point_at(results,hand_id,16,4,"Top")and is_point_at(results,hand_id,20,4,"Top")):
   if(is_point_at(results,hand_id,8,4,"Left") and is_point_at(results,hand_id,4,20,"Left")):
     print("The letter represented is E") 
     E_letter=1
  return E_letter

def letter_F(results,hand_id):
  F_letter=0
  if(is_point_at(results,hand_id,12,16,"Top") and is_point_at(results,hand_id,16,20,"Top")
  and is_point_at(results,hand_id,18,8,"Top")and is_point_at(results,hand_id,4,3,"Top")
  and is_point_at(results,hand_id,3,2,"Top")and is_point_at(results,hand_id,2,1,"Top")
  and is_point_at(results,hand_id,4,8,"Top")and is_point_at(results,hand_id,4,5,"Top")
  and is_point_at(results,hand_id,11,7,"Top")and is_point_at(results,hand_id,7,8,"Top")
  and is_point_at(results,hand_id,12,11,"Top")and is_point_at(results,hand_id,16,15,"Top")
  and is_point_at(results,hand_id,20,19,"Top") ):
   if(is_point_at(results,hand_id,4,7,"Left") and is_point_at(results,hand_id,3,8,"Left")
   and is_point_at(results,hand_id,8,9,"Left")and is_point_at(results,hand_id,12,16,"Left")
   and is_point_at(results,hand_id,16,20,"Left")and is_point_at(results,hand_id,8,12,"Left")):
     F_letter=1
     print("The letter represented is F") 
  return F_letter


def letter_H(results,hand_id):
  H_letter=0
  if(is_point_at(results,hand_id,8,12,"Top") and is_point_at(results,hand_id,7,11,"Top")
  and is_point_at(results,hand_id,16,20,"Top")and is_point_at(results,hand_id,4,16,"Top")
  and is_point_at(results,hand_id,5,9,"Top")and is_point_at(results,hand_id,20,0,"Top")):
   if(is_point_at(results,hand_id,0,5,"Left") and is_point_at(results,hand_id,0,9,"Left")
   and is_point_at(results,hand_id,0,20,"Left")and is_point_at(results,hand_id,0,16,"Left")):
     H_letter=1
     print("The letter represented is H") 
  return H_letter

def letter_L(results,hand_id):
  L_letter=0
  if(is_point_at(results,hand_id,8,7,"Top") and is_point_at(results,hand_id,6,5,"Top")
  and is_point_at(results,hand_id,4,3,"Top")and is_point_at(results,hand_id,2,1,"Top")
  and is_point_at(results,hand_id,1,0,"Top")and is_point_at(results,hand_id,11,12,"Top")
  and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,19,20,"Top")):
   if(is_point_at(results,hand_id,5,9,"Left") and is_point_at(results,hand_id,9,13,"Left")
    and is_point_at(results,hand_id,13,17,"Left")and is_point_at(results,hand_id,1,5,"Left")):
     L_letter=1
     print("The letter represented is L")

  return L_letter

def letter_O(results,hand_id):
  O_letter=0
  if(is_point_at(results,hand_id,7,8,"Top") and is_point_at(results,hand_id,7,4,"Top")
  and is_point_at(results,hand_id,6,7,"Top")and is_point_at(results,hand_id,5,0,"Top")
  and is_point_at(results,hand_id,1,0,"Top")and is_point_at(results,hand_id,4,3,"Top")
  and is_point_at(results,hand_id,3,2,"Top")and is_point_at(results,hand_id,1,0,"Top")):
   if(is_point_at(results,hand_id,5,6,"Left") and is_point_at(results,hand_id,6,7,"Left")
    and is_point_at(results,hand_id,7,8,"Left")and is_point_at(results,hand_id,8,4,"Left")
    and is_point_at(results,hand_id,0,1,"Left")):
     print("The letter represented is O") 
     O_letter=1
  return O_letter

#OPENED HAND
def reset_gesture(results,hand_id):
  found=0
  #hand, thumb
  if(is_point_at(results,hand_id,4,3,"Left") and is_point_at(results,hand_id,3,2,"Left")
  and is_point_at(results,hand_id,1,0,"Left")and is_point_at(results,hand_id,5,0,"Top")
  and is_point_at(results,hand_id,9,0,"Top")and is_point_at(results,hand_id,13,0,"Top")
  and is_point_at(results,hand_id,17,0,"Top")and is_point_at(results,hand_id,5,4,"Top")
  and is_point_at(results,hand_id,4,5,"Left")):
    #fingers
    #index
    if(is_point_at(results,hand_id,8,7,"Top")and is_point_at(results,hand_id,6,5,"Top")
    and is_point_at(results,hand_id,8,12,"Left")):
      #middle
      offset = 4
      if(is_point_at(results,hand_id,8+offset,7+offset,"Top")and is_point_at(results,hand_id,6+offset,5+offset,"Top")
      and is_point_at(results,hand_id,8+offset,12+offset,"Left")):
        #ring
        offset = 8
        if(is_point_at(results,hand_id,8+offset,7+offset,"Top")and is_point_at(results,hand_id,6+offset,5+offset,"Top")
        and is_point_at(results,hand_id,8+offset,12+offset,"Left")):
          #pinky
          offset = 12
          if(is_point_at(results,hand_id,8+offset,7+offset,"Top")and is_point_at(results,hand_id,6+offset,5+offset,"Top")):
            print("Reset gesture found") 
            found=1
  return found

#get letters
def logic_get_letter(results,hand_id):
  letter = ""
  if letter_A(results,hand_id):
    letter = "A"
  if letter_B(results,hand_id):
    letter = "B"
  if letter_C(results,hand_id):
    letter = "C"
  if letter_D(results,hand_id):
    letter = "D"
  if letter_E(results,hand_id):
    letter = "E"
  if letter_F(results,hand_id):
    letter = "F"
  if letter_H(results,hand_id):
    letter = "H"
  if letter_L(results,hand_id):
    letter = "L"
  if letter_O(results,hand_id):
    letter = "O"
  if reset_gesture(results,hand_id):
    letter = "RESET"
  return letter

# Check current sign gesture in right hand
# Reset gesture if reset sign or exced max_letters
def get_current_letter(results, max_letters):
  global current_letter
  global letters
  current_letter = ""
  multi_hand_label = get_multi_hand_label(results)
  for hand_id, label in multi_hand_label.items():
      if label == "Right":
        temp_letter = logic_get_letter(results,hand_id)
        if ((len(letters) >= max_letters) and (len(temp_letter) != 0)):
          current_letter = "RESET"
        else:
          current_letter = temp_letter

# Update letters string
# Reset if "RESET" letter found
def update_letters():
  global letters
  if len(current_letter) > 0:
    if (current_letter == "RESET"):
      letters = ""
    else:
      letters = letters + current_letter

#updates track progress frames and return if it is a track frame
def track_progress():
  global t_frames
  track = 0
  if(t_frames == (max_track_frames-1)):
    t_frames = 0
  else:
    if (t_frames == 0):
      track = 1
    t_frames = t_frames + 1
  return track

#DRAW INTERFACE
def draw_if(image):
  draw_progress(image)
  draw_word(image)
  draw_current_letter(image)

#draw progress feedback bar
def draw_progress(image):
  draw_background_progress_bar(image)
  draw_progress_bar(image)

def draw_background_progress_bar(image):
  # Define bar parameters
  bar_width = int(image.shape[1] * 0.8)
  bar_height = int(image.shape[0] * 0.05)
  bar_x = int((image.shape[1] - bar_width) / 2)
  bar_y = int(image.shape[0] - bar_height - 10)
  # Draw background bar
  cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

def draw_progress_bar(image):
  # Define bar parameters
  bar_width = int(image.shape[1] * 0.8)
  bar_height = int(image.shape[0] * 0.05)
  bar_x = int((image.shape[1] - bar_width) / 2)
  bar_y = int(image.shape[0] - bar_height - 10)
  progress = min(t_frames / max_track_frames, 1)
  filled_width = int(bar_width * progress)
  # Draw progress bar
  cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (204, 204, 0), -1)

# draw tracked word
def draw_word(image):
  global letters
  if len(letters) > 0:
    # Get the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(letters, font, font_scale, font_thickness)

    # Calculate the position to center the text
    text_x = (image.shape[1] - text_width) // 2
    text_y = text_height + 20  # Adjust the vertical position as needed
    # Draw text background bar
    text_background_bar_y = text_y - text_height - 10
    cv2.rectangle(image, (text_x - 10, text_background_bar_y), (text_x + text_width + 10, text_background_bar_y + text_height + baseline + 10), (0, 0, 0), -1)

    # Draw the word on the image
    cv2.putText(image, letters, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

# draw current detected letter in screen
def draw_current_letter(image):
  global current_letter
  if len(current_letter) > 0:
    # Get the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(current_letter, font, font_scale, font_thickness)

    # Calculate the position to center the text
    margin = 20
    text_x = image.shape[1] - text_width - margin
    text_y = text_height + margin  # Adjust the vertical position as needed
    # Draw text background bar
    text_background_bar_y = text_y - text_height - 10
    cv2.rectangle(image, (text_x - 10, text_background_bar_y), (text_x + text_width + 10, text_background_bar_y + text_height + baseline + 10), (204, 204, 0), -1)

    # Draw the word on the image
    cv2.putText(image, current_letter, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

# code body
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6) as hands:
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

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)

    #TEST CODE
    ###########################
    #add letters to draw
    get_current_letter(results, MAX_LETTERS)

    #update laters if it is the track frame
    is_track_frame = track_progress()
    if(is_track_frame):
      update_letters()
    
    # Add GUI
    draw_if(image)
    #END TEST CODE
    ###########################

    # Draw image
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()

# Destroy all the windows 
cv2.destroyAllWindows() 