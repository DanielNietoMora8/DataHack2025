import cv2
import os
import imutils

personName = 'Juan Jose'
dataPath = 'C:/Users/USUARIO/Desktop/Semillero IA/Datahack/Data'
personPath = os.path.join(dataPath, personName)

if not os.path.exists(personPath):
    print('Carpeta creada:', personPath)
    os.makedirs(personPath)

# Asegúrate de que el archivo realmente se llama así y está en la ruta correcta
videoPath = 'C:/Users/USUARIO/Desktop/Semillero IA/Datahack/VideoPrueba/JuanJoseHerrera1.mp4'
cap = cv2.VideoCapture(videoPath)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if faceClassif.empty():
    print("❌ No se pudo cargar el clasificador Haar.")
    exit()

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Fin del video o error al leer.")
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    # Parámetros ajustados para mayor sensibilidad
    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    print(f"[INFO] Frame {count}: {len(faces)} rostro(s) detectado(s)")

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        filename = os.path.join(personPath, f'rostro_{count}.jpg')
        cv2.imwrite(filename, rostro)
        print("💾 Guardado:", filename)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Detección de Rostros', frame)

    if cv2.waitKey(1) == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()

