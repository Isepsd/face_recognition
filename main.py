import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
camera = cv2.VideoCapture(0)

def detect_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    return faces

def draw_boxes(frame):
    for (x, y, w, h) in detect_faces(frame):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
        cv2.putText(frame, "Isep sopiandani", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        frame_with_boxes = draw_boxes(frame)
        cv2.imshow("Face AI", frame_with_boxes)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()

if __name__ == '__main__':
    main()
