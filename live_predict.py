import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

# Load your trained model
model = joblib.load("isl_model.pkl")

# Letter mapping: 0=A, 1=B, ... 25=Z
LETTERS = [chr(ord('A') + i) for i in range(26)]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Default: both hands empty (all zeros), matching training format
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    uses_two_hands = 0

    prediction_text = "No hand detected"

    if result.multi_hand_landmarks and result.multi_handedness:
        detected_hands = list(zip(result.multi_hand_landmarks, result.multi_handedness))
        uses_two_hands = 1 if len(detected_hands) == 2 else 0

        for hand_landmarks, handedness in detected_hands:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = []
            for lm in hand_landmarks.landmark:
                coords += [lm.x, lm.y, lm.z]

            label = handedness.classification[0].label  # "Left" or "Right"
            if label == "Left":
                left_hand = coords
            else:
                right_hand = coords

        # Build the feature row in the exact same column order as training
        features = [uses_two_hands] + left_hand + right_hand
        features = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        predicted_letter = LETTERS[prediction]

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(features)[0][prediction] * 100
            if confidence < 50:
                prediction_text = "Uncertain..."
            else:
                prediction_text = f"{predicted_letter} ({confidence:.1f}%)"
        else:
            prediction_text = predicted_letter

    cv2.putText(frame, prediction_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 3)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

    cv2.imshow("Live ISL Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()