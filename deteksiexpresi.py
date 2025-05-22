from fer import FER
import cv2

detector = FER()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
valid_emotions = ['happy', 'angry', 'neutral', 'surprise']

while True:
    ret, frame = cap.read()# buat baca frame dari video
    
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     # ubah frame ke grayscale (untuk deteksi wajah)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # buatdeteksi ekspresi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #kotak ijo
        face = frame[y:y+h, x:x+w]
        emotion, score = detector.top_emotion(face)
        if emotion in valid_emotions:
            if emotion == "happy":
                message = "wah lagi bahagia ni!"
            elif emotion == "angry":
                message = "Santai Bro!"
            elif emotion == "neutral":
                message = "B aja nih."
            elif emotion == "surprise":
                message = "wowwwww!"
            else:
                message = "Ekspresi tidak dikenali"

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'Emotion: {emotion}', (10, 40), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Score: {score*100:.2f}%', (10, 80), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Message: {message}', (10, 120), font, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
