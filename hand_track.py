import cv2
import mediapipe as mp
from tqdm import tqdm
import array
from time import sleep
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

frames = 0
lock1 = lock2 = lock3 = lock4 = lock5 = 0


#CONFIG
#########
working_rate = 20
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
#it returns the points x and y respectively to each ids that ranges from [0..21]
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



def letter_A(results,hand_id):
  A_letter=0
  if(is_point_at(results,hand_id,7,8,"Top") and is_point_at(results,hand_id,11,12,"Top")
  and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,19,20,"Top")
  and is_point_at(results,hand_id,4,3,"Top")and is_point_at(results,hand_id,2,1,"Top")
  and is_point_at(results,hand_id,4,6,"Top")and is_point_at(results,hand_id,4,10,"Top")
  and is_point_at(results,hand_id,4,14,"Top")and is_point_at(results,hand_id,4,18,"Top")
  and is_point_at(results,hand_id,10,11,"Top")and is_point_at(results,hand_id,14,15,"Top")
  and is_point_at(results,hand_id,18,19,"Top")and is_point_at(results,hand_id,6,7,"Top")):
   if(is_point_at(results,hand_id,4,8,"Left") and is_point_at(results,hand_id,3,7,"Left")
   and is_point_at(results,hand_id,2,6,"Left")and is_point_at(results,hand_id,2,5,"Left")
   and is_point_at(results,hand_id,4,8,"Left")and is_point_at(results,hand_id,8,12,"Left")
   and is_point_at(results,hand_id,12,16,"Left")and is_point_at(results,hand_id,16,20,"Left")
   and is_point_at(results,hand_id,3,7,"Left")and is_point_at(results,hand_id,4,6,"Left")):
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
   and is_point_at(results,hand_id,6,4,"Left") and is_point_at(results,hand_id,5,4,"Left")
   and is_point_at(results,hand_id,14,4,"Left")and is_point_at(results,hand_id,0,9,"Left")):
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
  and is_point_at(results,hand_id,1,0,"Top")and is_point_at(results,hand_id,14,15,"Top")
  and is_point_at(results,hand_id,10,11,"Top")and is_point_at(results,hand_id,7,6,"Top")):
   if(is_point_at(results,hand_id,3,4,"Left") and is_point_at(results,hand_id,5,9,"Left")
   and is_point_at(results,hand_id,9,13,"Left")and is_point_at(results,hand_id,13,17,"Left")
   and is_point_at(results,hand_id,8,4,"Left")and is_point_at(results,hand_id,4,16,"Left")
   and is_point_at(results,hand_id,3,4,"Left")and is_point_at(results,hand_id,2,1,"Left")):
     D_letter=1
     print("The letter represented is D") 
  return D_letter


def letter_E(results,hand_id):
  E_letter=0
  if(is_point_at(results,hand_id,7,8,"Top") and is_point_at(results,hand_id,11,12,"Top")
  and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,19,20,"Top")
  and is_point_at(results,hand_id,8,4,"Top")and is_point_at(results,hand_id,12,4,"Top")
  and is_point_at(results,hand_id,16,4,"Top")and is_point_at(results,hand_id,20,4,"Top")
  and is_point_at(results,hand_id,4,0,"Top")):
   if(is_point_at(results,hand_id,8,4,"Left") and is_point_at(results,hand_id,4,20,"Left")
  and is_point_at(results,hand_id,4,13,"Left")):
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

def letter_G(results,hand_id):
  G_letter=0
  if(is_point_at(results,hand_id,11,12,"Top") and is_point_at(results,hand_id,15,16,"Top")
  and is_point_at(results,hand_id,19,20,"Top")and is_point_at(results,hand_id,8,7,"Top")
  and is_point_at(results,hand_id,7,11,"Top")and is_point_at(results,hand_id,4,11,"Top")
  and is_point_at(results,hand_id,4,15,"Top")and is_point_at(results,hand_id,4,20,"Top")
  and is_point_at(results,hand_id,6,5,"Top")and is_point_at(results,hand_id,7,6,"Top")
  and is_point_at(results,hand_id,10,4,"Top")and is_point_at(results,hand_id,4,5,"Top") ):
   if(is_point_at(results,hand_id,3,12,"Left") and is_point_at(results,hand_id,12,16,"Left")
   and is_point_at(results,hand_id,16,20,"Left")and is_point_at(results,hand_id,4,12,"Left")):
     G_letter=1
     print("The letter represented is G") 
  return G_letter



def letter_H(results,hand_id):
  H_letter=0
  if(is_point_at(results,hand_id,8,12,"Top") and is_point_at(results,hand_id,7,11,"Top")
  and is_point_at(results,hand_id,16,20,"Top")and is_point_at(results,hand_id,4,16,"Top")
  and is_point_at(results,hand_id,5,9,"Top")and is_point_at(results,hand_id,16,20,"Top")
  and is_point_at(results,hand_id,12,11,"Top")and is_point_at(results,hand_id,11,10,"Top")
  and is_point_at(results,hand_id,8,7,"Top")and is_point_at(results,hand_id,7,6,"Top")
  and is_point_at(results,hand_id,6,5,"Top")):
   if(is_point_at(results,hand_id,0,5,"Left") and is_point_at(results,hand_id,0,9,"Left")
   and is_point_at(results,hand_id,0,13,"Left")and is_point_at(results,hand_id,0,17,"Left")
   and  is_point_at(results,hand_id,16,4,"Left")and is_point_at(results,hand_id,5,4,"Left")
   and is_point_at(results,hand_id,20,10,"Left")and is_point_at(results,hand_id,16,10,"Left")):
     H_letter=1
     print("The letter represented is H") 
  return H_letter

def letter_I(results,hand_id):
  I_letter=0
  if(is_point_at(results,hand_id,20,19,"Top") and is_point_at(results,hand_id,15,16,"Top")
  and is_point_at(results,hand_id,11,12,"Top")and is_point_at(results,hand_id,3,8,"Top")
  and is_point_at(results,hand_id,3,2,"Top")and is_point_at(results,hand_id,2,1,"Top")
  and is_point_at(results,hand_id,4,8,"Top")and is_point_at(results,hand_id,4,12,"Top")
  and is_point_at(results,hand_id,4,16,"Top")):
   if(is_point_at(results,hand_id,8,12,"Left") and is_point_at(results,hand_id,12,16,"Left")
   and is_point_at(results,hand_id,16,20,"Left")and is_point_at(results,hand_id,8,4,"Left")
   and is_point_at(results,hand_id,4,16,"Left")and is_point_at(results,hand_id,5,4,"Left")
   and is_point_at(results,hand_id,4,0,"Left")and is_point_at(results,hand_id,12,0,"Left")):
     I_letter=1
     print("The letter represented is I") 
  return I_letter

def letter_K(results,hand_id):
  K_letter=0
  if(is_point_at(results,hand_id,12,8,"Top") and is_point_at(results,hand_id,14,18,"Top")
  and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,19,20,"Top")
  and is_point_at(results,hand_id,4,14,"Top")and is_point_at(results,hand_id,4,5,"Top")
  and is_point_at(results,hand_id,5,15,"Top")and is_point_at(results,hand_id,5,19,"Top")
  and is_point_at(results,hand_id,8,7,"Top")and is_point_at(results,hand_id,7,6,"Top")
  and is_point_at(results,hand_id,6,5,"Top")and is_point_at(results,hand_id,12,11,"Top")
  and is_point_at(results,hand_id,11,10,"Top")):
   if(is_point_at(results,hand_id,8,4,"Left") and is_point_at(results,hand_id,7,4,"Left")
   and is_point_at(results,hand_id,6,4,"Left")and is_point_at(results,hand_id,4,12,"Left")
   and is_point_at(results,hand_id,4,14,"Left")):
     K_letter=1
     print("The letter represented is K") 
  return K_letter


def letter_L(results,hand_id):
  L_letter=0
  if(is_point_at(results,hand_id,8,7,"Top") and is_point_at(results,hand_id,6,5,"Top")
  and is_point_at(results,hand_id,4,3,"Top")and is_point_at(results,hand_id,2,1,"Top")
  and is_point_at(results,hand_id,1,0,"Top")and is_point_at(results,hand_id,11,12,"Top")
  and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,19,20,"Top")
  and is_point_at(results,hand_id,5,4,"Top")and is_point_at(results,hand_id,14,4,"Top")
  and is_point_at(results,hand_id,7,6,"Top")and is_point_at(results,hand_id,3,2,"Top")
  and is_point_at(results,hand_id,10,11,"Top")and is_point_at(results,hand_id,14,15,"Top")):
   if(is_point_at(results,hand_id,5,9,"Left") and is_point_at(results,hand_id,9,13,"Left")
    and is_point_at(results,hand_id,13,17,"Left")and is_point_at(results,hand_id,1,5,"Left")
    and is_point_at(results,hand_id,5,9,"Left")and is_point_at(results,hand_id,4,3,"Left")
    and is_point_at(results,hand_id,3,2,"Left")and is_point_at(results,hand_id,2,1,"Left")
    and is_point_at(results,hand_id,2,5,"Left")):
     L_letter=1
     print("The letter represented is L")
  return L_letter




def letter_M(results,hand_id):
  M_letter=0
  if(is_point_at(results,hand_id,14,4,"Top") and is_point_at(results,hand_id,14,18,"Top")
  and is_point_at(results,hand_id,6,7,"Top")and is_point_at(results,hand_id,7,8,"Top")
  and is_point_at(results,hand_id,10,4,"Top")and is_point_at(results,hand_id,14,4,"Top")
  and is_point_at(results,hand_id,6,4,"Top")and is_point_at(results,hand_id,11,12,"Top")
  and is_point_at(results,hand_id,14,15,"Top")and is_point_at(results,hand_id,15,16,"Top")
  and is_point_at(results,hand_id,18,19,"Top")and is_point_at(results,hand_id,19,20,"Top")
  and is_point_at(results,hand_id,18,20,"Top")and is_point_at(results,hand_id,8,20,"Top")
  and is_point_at(results,hand_id,12,20,"Top")and is_point_at(results,hand_id,4,18,"Top")
  and is_point_at(results,hand_id,16,20,"Top")):
   if(is_point_at(results,hand_id,16,4,"Left") and is_point_at(results,hand_id,8,4,"Left")
    and is_point_at(results,hand_id,14,4,"Left")and is_point_at(results,hand_id,8,12,"Left")
    and is_point_at(results,hand_id,12,16,"Left")and is_point_at(results,hand_id,16,20,"Left")
    and is_point_at(results,hand_id,6,10,"Left")and is_point_at(results,hand_id,10,14,"Left")
    and is_point_at(results,hand_id,14,18,"Left")):
     M_letter=1
     print("The letter represented is M")
  return M_letter


def letter_N(results,hand_id):
  N_letter=0
  if(is_point_at(results,hand_id,4,14,"Top") and is_point_at(results,hand_id,14,18,"Top")
  and is_point_at(results,hand_id,10,6,"Top")and is_point_at(results,hand_id,6,7,"Top")
  and is_point_at(results,hand_id,7,8,"Top")and is_point_at(results,hand_id,10,11,"Top")
  and is_point_at(results,hand_id,14,15,"Top")and is_point_at(results,hand_id,11,12,"Top")
  and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,15,16,"Top")
  and is_point_at(results,hand_id,18,19,"Top")and is_point_at(results,hand_id,19,20,"Top")
  and is_point_at(results,hand_id,12,8,"Top")and is_point_at(results,hand_id,8,16,"Top")
  and is_point_at(results,hand_id,4,10,"Top")):
   if(is_point_at(results,hand_id,12,4,"Left") and is_point_at(results,hand_id,8,4,"Left")
    and is_point_at(results,hand_id,10,4,"Left")and is_point_at(results,hand_id,4,14,"Left")
    and is_point_at(results,hand_id,8,12,"Left")and is_point_at(results,hand_id,12,16,"Left")
    and is_point_at(results,hand_id,16,20,"Left")and is_point_at(results,hand_id,6,10,"Left")
    and is_point_at(results,hand_id,10,14,"Left"))and is_point_at(results,hand_id,14,18,"Left"):
     N_letter=1
     print("The letter represented is N")
  return N_letter


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

def letter_P(results,hand_id):
  P_letter=0
  if(is_point_at(results,hand_id,8,7,"Top") and is_point_at(results,hand_id,7,6,"Top")
  and is_point_at(results,hand_id,6,5,"Top")and is_point_at(results,hand_id,6,10,"Top")
  and is_point_at(results,hand_id,10,14,"Top")and is_point_at(results,hand_id,15,16,"Top")
  and is_point_at(results,hand_id,5,9,"Top")and is_point_at(results,hand_id,9,13,"Top")
  and is_point_at(results,hand_id,13,17,"Top")and is_point_at(results,hand_id,4,14,"Top")):
   if(is_point_at(results,hand_id,5,4,"Left") and is_point_at(results,hand_id,4,6,"Left")
   and is_point_at(results,hand_id,4,10,"Left")and is_point_at(results,hand_id,4,16,"Left")
   and is_point_at(results,hand_id,0,5,"Left")and is_point_at(results,hand_id,0,9,"Left")
   and is_point_at(results,hand_id,0,13,"Left")and is_point_at(results,hand_id,0,17,"Left")):
     P_letter=1
     print("The letter represented is P") 
  return P_letter

def letter_Q(results,hand_id):
  Q_letter=0
  if(is_point_at(results,hand_id,6,7,"Top") and is_point_at(results,hand_id,7,8,"Top")
  and is_point_at(results,hand_id,5,6,"Top")and is_point_at(results,hand_id,3,4,"Top")
  and is_point_at(results,hand_id,2,3,"Top")and is_point_at(results,hand_id,1,2,"Top")
  and is_point_at(results,hand_id,0,17,"Top")and is_point_at(results,hand_id,5,1,"Top")
  and is_point_at(results,hand_id,10,6,"Top")):
   if(is_point_at(results,hand_id,1,2,"Left") and is_point_at(results,hand_id,2,3,"Left")
   and is_point_at(results,hand_id,3,4,"Left")and is_point_at(results,hand_id,5,6,"Left")
   and is_point_at(results,hand_id,9,10,"Left")and is_point_at(results,hand_id,13,14,"Left")
   and is_point_at(results,hand_id,17,18,"Left")and is_point_at(results,hand_id,0,11,"Left")):
     Q_letter=1
     print("The letter represented is Q") 
  return Q_letter


def letter_R(results,hand_id):
  R_letter=0
  if(is_point_at(results,hand_id,12,8,"Top") and is_point_at(results,hand_id,8,7,"Top")
  and is_point_at(results,hand_id,12,11,"Top")and is_point_at(results,hand_id,14,4,"Top")
  and is_point_at(results,hand_id,4,16,"Top")and is_point_at(results,hand_id,4,20,"Top")
  and is_point_at(results,hand_id,6,4,"Top")and is_point_at(results,hand_id,10,4,"Top")
  and is_point_at(results,hand_id,19,20,"Top")and is_point_at(results,hand_id,15,16,"Top")):
    if(is_point_at(results,hand_id,12,8,"Left") and is_point_at(results,hand_id,12,4,"Left")
    and is_point_at(results,hand_id,8,4,"Left")and is_point_at(results,hand_id,4,20,"Left")):
      R_letter=1
      print("The letter represented is R") 
  return R_letter

def letter_S(results,hand_id):
  S_letter=0
  if(is_point_at(results,hand_id,4,12,"Top")
  and is_point_at(results,hand_id,4,16,"Top")and is_point_at(results,hand_id,4,20,"Top")
  and is_point_at(results,hand_id,14,18,"Top")and is_point_at(results,hand_id,10,6,"Top")
  and is_point_at(results,hand_id,6,14,"Top")and is_point_at(results,hand_id,4,1,"Top")
  and is_point_at(results,hand_id,14,15,"Top")and is_point_at(results,hand_id,10,11,"Top")
  and is_point_at(results,hand_id,18,19,"Top")and is_point_at(results,hand_id,7,8,"Top")
  and is_point_at(results,hand_id,6,7,"Top")):
   if(is_point_at(results,hand_id,6,4,"Left") and is_point_at(results,hand_id,4,14,"Left")
    and is_point_at(results,hand_id,10,14,"Left")and is_point_at(results,hand_id,14,18,"Left")
    and is_point_at(results,hand_id,4,0,"Left")):
     S_letter=1
     print("The letter represented is S")
  return S_letter

def letter_T(results,hand_id):
  T_letter=0

  if(is_point_at(results,hand_id,4,3,"Top")and is_point_at(results,hand_id,2,1,"Top")
  and is_point_at(results,hand_id,4,6,"Top")and is_point_at(results,hand_id,4,10,"Top")
  and is_point_at(results,hand_id,4,14,"Top")and is_point_at(results,hand_id,4,18,"Top")
  and is_point_at(results,hand_id,10,11,"Top")and is_point_at(results,hand_id,14,15,"Top")
  and is_point_at(results,hand_id,18,19,"Top")and is_point_at(results,hand_id,6,7,"Top")):
   if(is_point_at(results,hand_id,6,4,"Left")and is_point_at(results,hand_id,4,10,"Left")
   and is_point_at(results,hand_id,10,14,"Left")and is_point_at(results,hand_id,14,18,"Left")):
     T_letter=1
     print("The letter represented is T") 
  return T_letter


def letter_U(results,hand_id):
  U_letter=0
  if(is_point_at(results,hand_id,12,11,"Top") and is_point_at(results,hand_id,11,10,"Top")
  and is_point_at(results,hand_id,10,9,"Top")and is_point_at(results,hand_id,15,16,"Top")
  and is_point_at(results,hand_id,19,20,"Top")and is_point_at(results,hand_id,8,7,"Top")
  and is_point_at(results,hand_id,4,15,"Top")and is_point_at(results,hand_id,4,19,"Top")
  and is_point_at(results,hand_id,6,5,"Top")and is_point_at(results,hand_id,7,6,"Top")):
   if(is_point_at(results,hand_id,9,4,"Left") and is_point_at(results,hand_id,16,4,"Left")
    and is_point_at(results,hand_id,2,8,"Left")and is_point_at(results,hand_id,8,12,"Left")
    and is_point_at(results,hand_id,14,18,"Left")
    and is_point_at(results,hand_id,5,8,"Left")):
     U_letter=1
     print("The letter represented is U")
  return U_letter

def letter_V(results,hand_id):
  V_letter=0
  if(is_point_at(results,hand_id,12,11,"Top") and is_point_at(results,hand_id,11,10,"Top")
  and is_point_at(results,hand_id,10,9,"Top")and is_point_at(results,hand_id,15,16,"Top")
  and is_point_at(results,hand_id,19,20,"Top")and is_point_at(results,hand_id,8,7,"Top")
  and is_point_at(results,hand_id,4,15,"Top")and is_point_at(results,hand_id,4,19,"Top")
  and is_point_at(results,hand_id,6,5,"Top")and is_point_at(results,hand_id,7,6,"Top")):
   if(is_point_at(results,hand_id,9,4,"Left") and is_point_at(results,hand_id,16,4,"Left")
    and is_point_at(results,hand_id,8,12,"Left")and is_point_at(results,hand_id,9,12,"Left")
    and is_point_at(results,hand_id,14,18,"Left")
    and is_point_at(results,hand_id,8,5,"Left")):
     V_letter=1
     print("The letter represented is V")
  return V_letter

def letter_W(results,hand_id):
  W_letter=0
  if(is_point_at(results,hand_id,12,11,"Top") and is_point_at(results,hand_id,11,10,"Top")
  and is_point_at(results,hand_id,10,9,"Top")and is_point_at(results,hand_id,16,15,"Top")
  and is_point_at(results,hand_id,19,20,"Top")and is_point_at(results,hand_id,8,7,"Top")
  and is_point_at(results,hand_id,14,13,"Top")and is_point_at(results,hand_id,4,19,"Top")
  and is_point_at(results,hand_id,6,5,"Top")and is_point_at(results,hand_id,7,6,"Top")):
   if(is_point_at(results,hand_id,9,4,"Left") and is_point_at(results,hand_id,4,16,"Left")
    and is_point_at(results,hand_id,8,12,"Left")
    and is_point_at(results,hand_id,9,14,"Left")):
     W_letter=1
     print("The letter represented is W")
  return W_letter

def letter_X(results,hand_id):
  X_letter=0
  if (is_point_at(results,hand_id,6,5,"Top")
  and is_point_at(results,hand_id,10,2,"Top")and is_point_at(results,hand_id,5,9,"Top")
  and is_point_at(results,hand_id,9,13,"Top")and is_point_at(results,hand_id,13,17,"Top")
  and is_point_at(results,hand_id,12,17,"Top")and is_point_at(results,hand_id,8,12,"Top")
  and is_point_at(results,hand_id,12,16,"Top")and is_point_at(results,hand_id,12,20,"Top")
  and is_point_at(results,hand_id,8,10,"Top")and is_point_at(results,hand_id,7,8,"Top")):
    if(is_point_at(results,hand_id,2,3,"Left")and is_point_at(results,hand_id,6,7,"Left")
    and is_point_at(results,hand_id,7,8,"Left")
    and is_point_at(results,hand_id,5,9,"Left")and is_point_at(results,hand_id,9,13,"Left")
    and is_point_at(results,hand_id,13,17,"Left")):
     X_letter=1
     print("The letter represented is X")
  return X_letter

def letter_Y(results,hand_id):
  Y_letter=0
  if(is_point_at(results,hand_id,6,4,"Top") and is_point_at(results,hand_id,4,3,"Top")
  and is_point_at(results,hand_id,3,2,"Top")and is_point_at(results,hand_id,2,1,"Top")
  and is_point_at(results,hand_id,20,19,"Top")and is_point_at(results,hand_id,19,18,"Top")
  and is_point_at(results,hand_id,18,17,"Top")and is_point_at(results,hand_id,7,8,"Top")
  and is_point_at(results,hand_id,11,12,"Top")and is_point_at(results,hand_id,15,16,"Top")
  and is_point_at(results,hand_id,5,8,"Top")and is_point_at(results,hand_id,13,15,"Top")
  and is_point_at(results,hand_id,9,11,"Top")):
   if(is_point_at(results,hand_id,4,3,"Left") and is_point_at(results,hand_id,3,2,"Left")
    and is_point_at(results,hand_id,2,1,"Left")and is_point_at(results,hand_id,1,8,"Left")
    and is_point_at(results,hand_id,8,12,"Left")and is_point_at(results,hand_id,12,16,"Left")
    and is_point_at(results,hand_id,16,20,"Left")and is_point_at(results,hand_id,0,20,"Left")):
     Y_letter=1
     print("The letter represented is Y")
  return Y_letter


#Defines an hearth made with both hands
def corazon(results,hand_id):
  right_hand=(0,0)
  left_hand=(0,0)
  multi_hand_label = get_multi_hand_label(results)
  for hand_id, label in multi_hand_label.items():
    if label == "Right":
     if(is_point_at(results,hand_id,7,8,"Top")and is_point_at(results,hand_id,6,7,"Top")and is_point_at(results,hand_id,6,5,"Top")
      and is_point_at(results,hand_id,11,12,"Top")and is_point_at(results,hand_id,10,11,"Top")and is_point_at(results,hand_id,10,9,"Top")
      and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,14,15,"Top")and is_point_at(results,hand_id,14,13,"Top")
      and is_point_at(results,hand_id,19,20,"Top")and is_point_at(results,hand_id,18,19,"Top")and is_point_at(results,hand_id,18,17,"Top")
      and is_point_at(results,hand_id,0,4,"Top")):
        if(is_point_at(results,hand_id,4,3,"Left") and is_point_at(results,hand_id,3,2,"Left")
        and is_point_at(results,hand_id,2,1,"Left")and is_point_at(results,hand_id,1,0,"Left")
        and is_point_at(results,hand_id,5,1,"Left")and is_point_at(results,hand_id,9,1,"Left")
        and is_point_at(results,hand_id,13,1,"Left")and is_point_at(results,hand_id,17,1,"Left")):
         right_hand=get_hand_point_landmark(results,hand_id,8)
    if label == "Left":
            if(is_point_at(results,hand_id,7,8,"Top")and is_point_at(results,hand_id,6,7,"Top")and is_point_at(results,hand_id,6,5,"Top")
              and is_point_at(results,hand_id,11,12,"Top")and is_point_at(results,hand_id,10,11,"Top")and is_point_at(results,hand_id,10,9,"Top")
              and is_point_at(results,hand_id,15,16,"Top")and is_point_at(results,hand_id,14,15,"Top")and is_point_at(results,hand_id,14,13,"Top")
              and is_point_at(results,hand_id,19,20,"Top")and is_point_at(results,hand_id,18,19,"Top")and is_point_at(results,hand_id,18,17,"Top")):
              if(is_point_at(results,hand_id,3,4,"Left") and is_point_at(results,hand_id,2,3,"Left")
                and is_point_at(results,hand_id,1,2,"Left")and is_point_at(results,hand_id,0,1,"Left")
                and is_point_at(results,hand_id,1,5,"Left")and is_point_at(results,hand_id,1,9,"Left")
                and is_point_at(results,hand_id,1,13,"Left")and is_point_at(results,hand_id,1,17,"Left")):
                   left_hand=get_hand_point_landmark(results,hand_id,8)
  if(((right_hand[1]-left_hand[1])<0.2) and ((right_hand[0]-left_hand[0])<0.2) and right_hand!=(0,0)and left_hand!=(0,0)):
    print("<3")           

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  progress_bar = tqdm(total=5, desc='HELLO', position=0, leave=True)
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
      multi_hand_label = get_multi_hand_label(results)
      for hand_id, label in multi_hand_label.items():
        if label == "Right":
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
            
            #Just to test if the letters are well printed
            letter_A(results, hand_id)
            letter_B(results, hand_id)
            letter_C(results, hand_id)
            letter_D(results, hand_id)
            letter_E(results, hand_id)
            letter_F(results, hand_id)
            letter_G(results, hand_id)
            letter_H(results, hand_id)            
            letter_I(results, hand_id)
            letter_K(results, hand_id)                       
            letter_L(results, hand_id)
            letter_M(results, hand_id)
            letter_N(results, hand_id)
            letter_O(results, hand_id)
            letter_P(results, hand_id)
            letter_Q(results, hand_id)
            letter_R(results, hand_id)
            letter_S(results,hand_id)
            letter_T(results,hand_id)
            letter_U(results,hand_id)
            letter_X(results,hand_id)
            letter_W(results,hand_id)
            letter_V(results,hand_id)
            letter_Y(results,hand_id)
            corazon(results,hand_id)
#Attemtives to make it simplier
        #detect_hello(label,lock1,lock2,lock3,lock4,lock5)
        #letter_E(results, hand_id)   
          #L1 = letter_H(results, hand_id)
          #L2 = letter_E(results, hand_id)
          #L3 = letter_L(results, hand_id)
          #L4 = letter_L(results, hand_id)
          #L5 = letter_O(results, hand_id)
          #arr_hello=array.array('i',[L1,L2,L3,L4,L5])
          #progress_bar(arr_hello)
          

          #letter_H()
          #if(is_point_at(results, hand_id, 4, 8, "Left")):
            #print("On right hand, thumb is on the left of index finger")
          #if(is_point_at(results,hand_id,8,0,"Top")):
            #print("thumb is above the wrist") 


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
