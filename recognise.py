import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("Trained_model.h5")

image_x, image_y = 64, 64

classes = [
    'Be My Friend', 'Do not Worry', 'Get well soon',
    'Give Me Water', 'Good Morning', 'Good Night',
    'Help Me', 'How are You', 'Hungry', 'Hurry Up',
    'I am Fine', 'I am Late', 'I am Not Feeling Well',
    'I am Tired', 'I Love Reading', 'I Love Writing',
    'I will Try my level Best', 'Its Raining Outside',
    'Lets Dance', 'Lets Go', 'Lets Play',
    'Long Time no See', 'Look Down', 'Look Up',
    'Medicine', 'Music'
]

cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Draw rectangle
    cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), 2)

    roi = frame[102:298, 427:623]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 10, 60])
    upper = np.array([20, 150, 255])

    mask = cv2.inRange(hsv, lower, upper)

    resized = cv2.resize(mask, (image_x, image_y))

    # Convert to 3 channel
    mask_3ch = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    normalized = mask_3ch / 255.0
    reshaped = normalized.reshape(1, image_x, image_y, 3)

    prediction = model.predict(reshaped, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence > 0.75:
        text = f"{classes[class_index]} ({confidence:.2f})"
    else:
        text = "Detecting..."

    cv2.putText(frame, text, (30, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
