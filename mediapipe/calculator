import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

right_number = None
left_op = None
numbers = []
operation = None
result = None
last_right = None
last_left = None
gesture_time_right = 0
gesture_time_left = 0

op_dict = {1: "+", 2: "-", 3: "*", 4: "/"}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    right_fingers = 0
    left_fingers = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # "Left" o "Right"
            tips_ids = [4, 8, 12, 16, 20]
            fingers_up = []

            # Pulgar
            if hand_label == "Right":
                if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            else:  # Left
                if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # Otros dedos
            for id in range(1, 5):
                if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            count = sum(fingers_up)
            if hand_label == "Right":
                right_fingers = count
            else:
                left_fingers = count

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- Lógica para mano derecha (números) ---
    if right_fingers != last_right:
        last_right = right_fingers
        gesture_time_right = time.time()
    else:
        if time.time() - gesture_time_right >= 2 and 1 <= right_fingers <= 5:
            if operation is None:
                if len(numbers) == 0 or (len(numbers) == 1 and result is not None):
                    numbers = [right_fingers]
                    result = None
            elif len(numbers) == 1:
                numbers.append(right_fingers)
            gesture_time_right = time.time()

    # --- Lógica para mano izquierda (operaciones y AC) ---
    if left_fingers != last_left:
        last_left = left_fingers
        gesture_time_left = time.time()
    else:
        if time.time() - gesture_time_left >= 2:
            if left_fingers == 0:
                numbers = []
                operation = None
                result = None
            elif left_fingers in op_dict and len(numbers) == 1:
                operation = op_dict[left_fingers]
            gesture_time_left = time.time()

    # --- Calcular resultado ---
    if len(numbers) == 2 and operation:
        a, b = numbers
        try:
            if operation == "+":
                result = a + b
            elif operation == "-":
                result = a - b
            elif operation == "*":
                result = a * b
            elif operation == "/":
                result = round(a / b, 2) if b != 0 else "Error"
        except Exception:
            result = "Error"
        numbers = []
        operation = None

    # --- Mostrar en pantalla ---
    cv2.putText(image, f"Numero: {numbers[0] if numbers else ''}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(image, f"Operacion: {operation if operation else ''}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
    if len(numbers) == 2:
        cv2.putText(image, f"Numero2: {numbers[1]}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    if result is not None:
        cv2.putText(image, f"Resultado: {result}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    cv2.putText(image, f"Mano D: {right_fingers}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 2)
    cv2.putText(image, f"Mano I: {left_fingers}", (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (128,0,255), 2)

    cv2.imshow("Calculadora con Gestos", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()