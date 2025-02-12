import cv2
import os

cascade_path = r"C:\Users\User\PycharmProjects\bill\pythonProject1\haarcascade_car.xml"

if not os.path.exists(cascade_path):
    print(f"Error: Cascade file not found at {cascade_path}")
    exit()

car_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

cv2.namedWindow("Car Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Car Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
