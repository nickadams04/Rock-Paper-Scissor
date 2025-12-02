import cv2

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

# Force MJPEG (very important)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Safe resolution & FPS for WSL
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    print("ret:", ret)
    if not ret:
        break

    cv2.imshow("WSL Camera", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
