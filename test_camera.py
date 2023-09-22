import cv2
cap = cv2.VideoCapture("rtsp://admin:Hikvision07!@192.169.0.100:554/h264/ch1/main/av_stream")

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
