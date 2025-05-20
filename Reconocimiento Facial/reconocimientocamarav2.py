import cv2
import os
import imutils

dataPath = 'C:/Users/USUARIO/Desktop/Semillero IA/Datahack/Data'
imagePaths = os.listdir(dataPath)
print('Personas detectadas en el modelo:', imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if faceClassif.empty():
    print("❌ No se pudo cargar el clasificador Haar.")
    exit()

print("🎥 Iniciando cámara. Presiona ESC para salir.")

# Bandera para saber si ya se reconoció una persona
persona_reconocida = False
nombre_reconocido = ''

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo acceder a la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        if not persona_reconocida:
            label, confidence = face_recognizer.predict(rostro)
            if confidence < 70:
                nombre_reconocido = imagePaths[label]
                persona_reconocida = True
                print(f"✅ Reconocido como: {nombre_reconocido}")
                color = (0, 255, 0)
            else:
                nombre_reconocido = 'Desconocido'
                print("❌ Rostro no reconocido.")
                color = (0, 0, 255)
        else:
            # Ya hay alguien reconocido, pero si ahora aparece alguien no reconocido
            label, confidence = face_recognizer.predict(rostro)
            if confidence >= 70:
                print("❌ Apareció un rostro no reconocido. Reanudando predicción...")
                persona_reconocida = False
                nombre_reconocido = ''
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

        if nombre_reconocido != '':
            cv2.putText(frame, f'{nombre_reconocido}', (x, y - 10), 2, 0.8, color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('Reconocimiento Facial en Vivo', frame)

    if cv2.waitKey(1) == 27:
        print("🛑 Reconocimiento detenido por el usuario.")
        break

cap.release()
cv2.destroyAllWindows()

