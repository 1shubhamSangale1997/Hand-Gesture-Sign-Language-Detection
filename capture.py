import cv2

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv2.imshow("Capture Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite("captured_image.jpg", frame)
        print("Image Captured!")
        break

cam.release()
cv2.destroyAllWindows()
