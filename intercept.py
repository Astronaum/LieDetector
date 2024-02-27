#importation des outils necessaires au fonctionnement du code

import argparse
import cv2
import mediapipe as mp
from ffpyplayer.player import MediaPlayer
from datetime import datetime
from matplotlib import pyplot as plt
import mss
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import distance as dist
from fer import FER
import threading
import time
import sys

#Partie declaration et initailisation
MAX_FRAMES = 120
RECENT_FRAMES = int(MAX_FRAMES / 10)
EYE_BLINK_HEIGHT = .15
SIGNIFICANT_BPM_CHANGE = 8
LIP_COMPRESSION_RATIO = .35
TELL_MAX_TTL = 30
TEXT_HEIGHT = 30
FACEMESH_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
EPOCH = time.time()
recording = None 
tells = dict()
blinks = [False] * MAX_FRAMES
blinks2 = [False] * MAX_FRAMES # Pour l'effet mirroir
hand_on_face = [False] * MAX_FRAMES
hand_on_face2 = [False] * MAX_FRAMES # Pour l'effet mirroir
face_area_size = 0 
hr_times = list(range(0, MAX_FRAMES))
hr_values = [400] * MAX_FRAMES
avg_bpms = [0] * MAX_FRAMES
gaze_values = [0] * MAX_FRAMES
emotion_detector = FER(mtcnn=True)
calculating_mood = False
mood = ''
meter = cv2.imread('meter.png')
# BPM chart
fig = None
ax = None
line = None
peakpts = None


# initialisation de la charte concernant l'affichage des battement du coeur si c'est indiquée au niveau du lancement
def chart_setup():
  global fig, ax, line, peakpts

  plt.ion()
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1) # 1er 1x1 subplot
  ax.set(ylim=(185, 200))
  line, = ax.plot(hr_times, hr_values, 'b-')
  peakpts, = ax.plot([], [], 'r+')

#decrementation des ttl
def decrement_tells(tells):
  for key, tell in tells.copy().items():
    if 'ttl' in tell:
      tell['ttl'] -= 1
      if tell['ttl'] <= 0:
        del tells[key]
  return tells

# Fonction du menu d'argument
def main():
  global TELL_MAX_TTL
  global recording

  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', nargs='*', help='Input video device (number or path), file, or screen dimensions (x y width height), defaults to 0', default=['0'])
  parser.add_argument('--landmarks', '-l', help='Set to any value to draw face and hand landmarks')
  parser.add_argument('--bpm', '-b', help='Set to any value to draw color chart for heartbeats')
  parser.add_argument('--flip', '-f', help='Set to any value to flip resulting output (selfie view)')
  parser.add_argument('--ttl', '-t', help='How many frames for each displayed "tell" to last, defaults to 30', default='30')
  parser.add_argument('--record', '-r', help='Set to any value to save a timestamped AVI in current directory')
  args = parser.parse_args()

  if len(args.input) == 1:
    INPUT = int(args.input[0]) if args.input[0].isdigit() else args.input[0]
  elif len(args.input) != 4:
    return print("Wrong number of values for 'input' argument; should be 0, 1, or 4.")

  DRAW_LANDMARKS = args.landmarks is not None
  BPM_CHART = args.bpm is not None
  FLIP = args.flip is not None
  if args.ttl and args.ttl.isdigit():
    TELL_MAX_TTL = int(args.ttl)
  RECORD = args.record is not None

  # Lancement du plot de BPM si defini
  if BPM_CHART:
    chart_setup()
    
  calibrated = False
  calibration_frames = 0
  with mp.solutions.face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    with mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7) as hands:
      if len(args.input) == 4:
        screen = {
          "top": int(args.input[0]),
          "left": int(args.input[1]),
          "width": int(args.input[2]),
          "height": int(args.input[3])
        }
        with mss.mss() as sct: # screenshot si on utilise une video enregistrée dans la machine 
          while True:
            image = np.array(sct.grab(screen))[:, :, :3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            calibration_frames += process(image, face_mesh, hands, calibrated, DRAW_LANDMARKS, BPM_CHART, FLIP)
            calibrated = (calibration_frames >= MAX_FRAMES)
            cv2.imshow('face', image)
            if RECORD:
              recording.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      else:
        cap = cv2.VideoCapture(INPUT) #Lancement de la capture de video
        fps = None
        if isinstance(INPUT, str) and INPUT.find('.') > -1: # from file
          fps = cap.get(cv2.CAP_PROP_FPS)
          print("FPS:", fps)
          # cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        else: # from device
          cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
          cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
          cap.set(cv2.CAP_PROP_FPS, 30)

        if RECORD: # Lancement de l'application d'après une video
          RECORDING_FILENAME = str(datetime.now()).replace('.','').replace(':','') + '.avi'
          FPS_OUT = 10
          FRAME_SIZE = (int(cap.get(3)), int(cap.get(4)))
          recording = cv2.VideoWriter(
            RECORDING_FILENAME, cv2.VideoWriter_fourcc(*'MJPG'), FPS_OUT, FRAME_SIZE)

        while cap.isOpened():
          success, image = cap.read()
          if not success: break
          calibration_frames += process(image, face_mesh, hands, calibrated, DRAW_LANDMARKS, BPM_CHART, FLIP, fps)
          calibrated = (calibration_frames >= MAX_FRAMES)
          cv2.imshow('face', image)
          if RECORD:
            recording.write(image)
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cap.release()
        if RECORD:
          recording.release()
  cv2.destroyAllWindows()

# tell = la donnée à capturer
def new_tell(result):
  global TELL_MAX_TTL

  return {
    'text': result,
    'ttl': TELL_MAX_TTL
  }

#definition de la fonction draw_on_frame qui trace le maillage de visage et des mains a base de la bibliotheque mp
def draw_on_frame(image, face_landmarks, hands_landmarks):
  mp.solutions.drawing_utils.draw_landmarks(
      image,
      face_landmarks,
      mp.solutions.face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_contours_style())
  mp.solutions.drawing_utils.draw_landmarks(
      image,
      face_landmarks,
      mp.solutions.face_mesh.FACEMESH_IRISES,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_iris_connections_style())
  for hand_landmarks in (hands_landmarks or []):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())

#Fonction d'affichage du mood (happy, sad..) et des autres metriques (frequences cardiaques, clignotement des paupieres,...) 
def add_text(image, tells, calibrated): 
  global mood

  text_y = TEXT_HEIGHT
  if mood:
    write("Mood: {}".format(mood), image, int(.75 * image.shape[1]), TEXT_HEIGHT)
  if calibrated:
    for tell in tells.values():
      write(tell['text'], image, 10, text_y)
      text_y += TEXT_HEIGHT


def write(text, image, x, y):
  cv2.putText(img=image, text=text, org=(x, y),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 0],
    lineType=cv2.LINE_AA, thickness=4)
  cv2.putText(img=image, text=text, org=(x, y),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 255, 255],
    lineType=cv2.LINE_AA, thickness=2)

#Fonction qui calcule ration d'ouverture bouche | oeuil
def get_aspect_ratio(top, bottom, right, left):
  height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
  width = dist.euclidean([right.x, right.y], [left.x, left.y])
  return height / width

# fonction pour avoir la surface 
def get_area(image, draw, topL, topR, bottomR, bottomL):
  topY = int((topR.y+topL.y)/2 * image.shape[0])
  botY = int((bottomR.y+bottomL.y)/2 * image.shape[0])
  leftX = int((topL.x+bottomL.x)/2 * image.shape[1])
  rightX = int((topR.x+bottomR.x)/2 * image.shape[1])

  if draw:
    image = cv2.circle(image, (leftX,topY), 2, (255,0,0), 2)
    image = cv2.circle(image, (leftX,botY), 2, (255,0,0), 2)
    image = cv2.circle(image, (rightX,topY), 2, (255,0,0), 2)
    image = cv2.circle(image, (rightX,botY), 2, (255,0,0), 2)

  return image[topY:botY, rightX:leftX]

#Fonction qui ajouter le bpm capturé à l'ensemble des données TELLS
def get_bpm_tells(cheekL, cheekR, fps, bpm_chart):
  global hr_times, hr_values, avg_bpms
  global ax, line, peakpts

  cheekLwithoutBlue = np.average(cheekL[:, :, 1:3])
  cheekRwithoutBlue = np.average(cheekR[:, :, 1:3])
  hr_values = hr_values[1:] + [cheekLwithoutBlue + cheekRwithoutBlue]

  if not fps:
    hr_times = hr_times[1:] + [time.time() - EPOCH]

  if bpm_chart:
    line.set_data(hr_times, hr_values)
    ax.relim()
    ax.autoscale()

  peaks, _ = find_peaks(hr_values,
    threshold=.1,
    distance=5,
    prominence=.5,
    wlen=10,
  )

  peak_times = [hr_times[i] for i in peaks]

  if bpm_chart:
    peakpts.set_data(peak_times, [hr_values[i] for i in peaks])

  bpms = 60 * np.diff(peak_times) / (fps or 1)
  bpms = bpms[(bpms > 50) & (bpms < 150)]  # filter les changement en BPM avec une portée raisonnable
  recent_bpms = bpms[(-3 * RECENT_FRAMES):]  # HR un long signal par rapport au autre

  recent_avg_bpm = 0
  bpm_display = "BPM: ..."
  if recent_bpms.size > 1:
    recent_avg_bpm = int(np.average(recent_bpms))
    bpm_display = "BPM: {} ({})".format(recent_avg_bpm, len(recent_bpms))

  avg_bpms = avg_bpms[1:] + [recent_avg_bpm]

  bpm_delta = 0
  bpm_change = ""

  if len(recent_bpms) > 2:
    all_bpms = list(filter(lambda bpm: bpm != '-', avg_bpms))
    all_avg_bpm = sum(all_bpms) / len(all_bpms)
    avg_recent_bpm = sum(recent_bpms) / len(recent_bpms)
    bpm_delta = avg_recent_bpm - all_avg_bpm

    if bpm_delta > SIGNIFICANT_BPM_CHANGE:
      bpm_change = "Heart rate increasing"
    elif bpm_delta < -SIGNIFICANT_BPM_CHANGE:
      bpm_change = "Heart rate decreasing"

  return bpm_display, bpm_change

# Verifier si oui on non il y a un clin d'oeil 
def is_blinking(face):
  eyeR = [face[p] for p in [159, 145, 133, 33]] #emplacement oeil droite sur le visage
  eyeR_ar = get_aspect_ratio(*eyeR) # aspect de l'oeil 

  eyeL = [face[p] for p in [386, 374, 362, 263]] # emplacement de l'oeil gauche sur visage 
  eyeL_ar = get_aspect_ratio(*eyeL) # aspect de l'oeil

  eyeA_ar = (eyeR_ar + eyeL_ar) / 2 # aspect des deux yeux 
  return eyeA_ar < EYE_BLINK_HEIGHT 

# Ajouter aux tells s'il y a un clin d'oeil
def get_blink_tell(blinks):
  if sum(blinks[:RECENT_FRAMES]) < 3: # le nombre de clins d'oeil est insuffisant pour faire une comparison
    return None

  recent_closed = 1.0 * sum(blinks[-RECENT_FRAMES:]) / RECENT_FRAMES # pourcentage des blink par rapport au frame ecent
  avg_closed = 1.0 * sum(blinks) / MAX_FRAMES # moyenne des blinks

  if recent_closed > (20 * avg_closed):
    return "Increased blinking"
  elif avg_closed >  (20 * recent_closed):
    return "Decreased blinking"
  else:
    return None


#determiner si la main se trouve à l'interieur du contour du visage 
def check_hand_on_face(hands_landmarks, face):
  if hands_landmarks:
    face_landmarks = [face[p] for p in FACEMESH_FACE_OVAL] # extraire les facemarks
    face_points = [[[p.x, p.y] for p in face_landmarks]] # extraire les face_point
    face_contours = np.array(face_points).astype(np.single) 

    for hand_landmarks in hands_landmarks:
      hand = []
      for point in hand_landmarks.landmark:
        hand.append( (point.x, point.y) )

      for finger in [4, 8, 20]:
        overlap = cv2.pointPolygonTest(face_contours, hand[finger], False)
        if overlap != -1:
          return True
  return False


#avoir la moyenne des directions du regard
def get_avg_gaze(face):
  gaze_left = get_gaze(face, 476, 474, 263, 362)
  gaze_right = get_gaze(face, 471, 469, 33, 133)
  return round((gaze_left + gaze_right) / 2, 1)

#avoir la direction du regard
def get_gaze(face, iris_L_side, iris_R_side, eye_L_corner, eye_R_corner):
  iris = (
    face[iris_L_side].x + face[iris_R_side].x,
    face[iris_L_side].y + face[iris_R_side].y,
  )
  eye_center = (
    face[eye_L_corner].x + face[eye_R_corner].x,
    face[eye_L_corner].y + face[eye_R_corner].y,
  )

  gaze_dist = dist.euclidean(iris, eye_center)
  eye_width = abs(face[eye_R_corner].x - face[eye_L_corner].x)
  gaze_relative = gaze_dist / eye_width

  if (eye_center[0] - iris[0]) < 0: # flip along x for looking L vs R
    gaze_relative *= -1

  return gaze_relative

#detecter le changement au niveau du regard
def detect_gaze_change(avg_gaze):
  global gaze_values

  gaze_values = gaze_values[1:] + [avg_gaze]
  gaze_relative_matches = 1.0 * gaze_values.count(avg_gaze) / MAX_FRAMES
  if gaze_relative_matches < .01: # looking in a new direction
    return gaze_relative_matches
  return 0

#calcul la ratio d'ouverture de la bouche
def get_lip_ratio(face):
  return get_aspect_ratio(face[0], face[17], face[61], face[291])

#detecter la mine
def get_mood(image):
  global emotion_detector, calculating_mood, mood

  detected_mood, score = emotion_detector.top_emotion(image)
  calculating_mood = False
  if score and (score > .4 or detected_mood == 'neutral'):
    mood = detected_mood

#ajouter une mesure de verite a la liste 

def add_truth_meter(image, tell_count):
  width = image.shape[1]
  sm = int(width / 64)
  bg = int(width / 3.2)

  resized_meter = cv2.resize(meter, (bg,sm), interpolation=cv2.INTER_AREA)
  image[sm:(sm+sm), bg:(bg+bg), 0:3] = resized_meter[:, :, 0:3]

  if tell_count:
    tellX = bg + int(bg/4) * (tell_count - 1) # adjust for always-on BPM
    cv2.rectangle(image, (tellX, int(.9*sm)), (tellX+int(sm/2), int(2.1*sm)), (0,0,0), 2)


#Fonction pour determiner la largeur et la longuer du visage

def get_face_relative_area(face):
  face_width = abs(max(face[454].x, 0) - max(face[234].x, 0))
  face_height = abs(max(face[152].y, 0) - max(face[10].y, 0))
  return face_width * face_height

#fonction qui localise le visage et les mains sur le frame
def find_face_and_hands(image_original, face_mesh, hands):
  image = image_original.copy()
  image.flags.writeable = False # pass by reference to improve speed
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Conversion de l'image de l'espace de couleur BGR a  l'espace RGB.

  faces = face_mesh.process(image)
  hands_landmarks = hands.process(image).multi_hand_landmarks

  face_landmarks = None
  if faces.multi_face_landmarks and len(faces.multi_face_landmarks) > 0:
    face_landmarks = faces.multi_face_landmarks[0] # use first face found

  return face_landmarks, hands_landmarks

# la fonction process s'occupe de rassembler l'ensemble des valeurs retournées par les cinq fonction et faire le calcule avec l'affichage des resultats sur truth meter

def process(image, face_mesh, hands, calibrated=False, draw=False, bpm_chart=False, flip=False, fps=None):
  global tells, calculating_mood
  global blinks, hand_on_face, face_area_size

  tells = decrement_tells(tells)

  face_landmarks, hands_landmarks = find_face_and_hands(image, face_mesh, hands)
  if face_landmarks:
    face = face_landmarks.landmark
    face_area_size = get_face_relative_area(face)

    if not calculating_mood:
      emothread = threading.Thread(target=get_mood, args=(image,))
      emothread.start()
      calculating_mood = True

    #Check cheek visibility?
    cheekL = get_area(image, draw, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
    cheekR = get_area(image, draw, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])

    avg_bpms, bpm_change = get_bpm_tells(cheekL, cheekR, fps, bpm_chart)
    tells['avg_bpms'] = new_tell(avg_bpms)  # toujours afficher "..." si BPM est innexistant
    if len(bpm_change):
      tells['bpm_change'] = new_tell(bpm_change)

    # Clins d'oeil : si il y a un changement dans le nbr de blink un tellest ajoutée dans la liste des tell
    
    blinks = blinks[1:] + [is_blinking(face)]
    recent_blink_tell = get_blink_tell(blinks)
    if recent_blink_tell:
      tells['blinking'] = new_tell(recent_blink_tell)

    # main sur visage : si la main se trouve sur le visage , on ajoute un tell
    
    recent_hand_on_face = check_hand_on_face(hands_landmarks, face)
    hand_on_face = hand_on_face[1:] + [recent_hand_on_face]
    if recent_hand_on_face:
      tells['hand'] = new_tell("Hand covering face")

    # suivie du regard : le changement de regard implique l'ajout d'un tell

    avg_gaze = get_avg_gaze(face)
    if detect_gaze_change(avg_gaze):
      tells['gaze'] = new_tell("Change in gaze")

    # compression des levres: si il y ' un une compression des levres un tell est ajouter à la liste
    
    if get_lip_ratio(face) < LIP_COMPRESSION_RATIO:
      tells['lips'] = new_tell("Lip compression")

    if bpm_chart: # mise à jour de la charte qui affiche les bPM
      fig.canvas.draw()
      fig.canvas.flush_events()

    if draw:  # recouvrir visage et mains landmark, 
      draw_on_frame(image, face_landmarks, hands_landmarks)

  if flip:
    image = cv2.flip(image, 1)  # retourner image horizontallement(effet selfie)

    #ajouter les valeurs trouver en haut a droite des frames
    
  add_text(image, tells, calibrated)
  
    # calculer le pourcentage de monsenge en fonction de la longueur de la liste des tell ,plus la liste est longue veut dire qu_il ya plus de chance que la personne mente 
    
    
  add_truth_meter(image, len(tells))

  return 1 if (face_landmarks and not calibrated) else 0

# fonction pour effectuer les comparaison

def mirror_compare(first, second, rate, less, more):
  if (rate * first) < second:
    return less
  elif first > (rate * second):
    return more
  return None

# detecter le changement des clins d'oeil


def get_blink_comparison(blinks1, blinks2):
  return mirror_compare(sum(blinks1), sum(blinks2), 1.8, "Blink less", "Blink more")

# detecter la comparaison des main couvrant visage


def get_hand_face_comparison(hand1, hand2):
  return mirror_compare(sum(hand1), sum(hand2), 2.1, "Stop touching face", "Touch face more")


# detecter comparaison des surfaces de visage 


def get_face_size_comparison(ratio1, ratio2):
  return mirror_compare(ratio1, ratio2, 1.5, "Too close", "Too far")




if __name__ == '__main__':
    main()
