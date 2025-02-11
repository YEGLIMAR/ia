import cv2
from ultralytics import YOLO

DROIDCAM_IP = "192.168.10.10"  
DROIDCAM_PORT = "4747"

model = YOLO("runs/detect/train7/weights/best.pt")

cap = cv2.VideoCapture(f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video")

if not cap.isOpened():
    print("❌ No se pudo conectar a DroidCam. Verifica la IP y el puerto.")
    exit()

print("✅ Conectado a DroidCam. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ No se recibió video. Revisa la conexión con DroidCam.")
        break


    results = model(frame)
    annotated_frame = results[0].plot()

    
    cv2.imshow("Detección en Tiempo Real - DroidCam", annotated_frame)

    # 📌 Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
