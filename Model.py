from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO("runs/detect/train7/weights/best.pt")

# Capturar video desde la cámara de la laptop (cambiar a 1 si tienes una segunda cámara)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Cámara detectada. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección con YOLO
    results = model(frame)

    # Dibujar las detecciones sobre la imagen
    annotated_frame = results[0].plot()

    # Mostrar la imagen con las detecciones
    cv2.imshow("Detección en Tiempo Real", annotated_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()

