import cv2
from cvzone.PoseModule import PoseDetector
import mediapipe as mp

face_mesh = mp.solutions.face_mesh
face_drawings = mp.solutions.drawing_utils
facemesh = mp.solutions.face_mesh.FaceMesh()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

detector = PoseDetector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = detector.findPose(frame)
    lmlist, bboxinfo = detector.findPosition(frame)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    result = facemesh.process(frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            face_drawings.draw_landmarks(frame, face_landmarks, face_mesh.FACEMESH_CONTOURS,
                                         face_drawings.DrawingSpec((0, 255, 0), 1, 1))

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

    cv2.imshow('Input', frame)
    # k = cv2.waitKey(160)
    c = cv2.waitKey(1) & 0xff
    if c == 27:
        break

    if not ret:
        print("failed to grab frame")
        break

cap.release()
cv2.destroyAllWindows()
