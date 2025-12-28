import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Carrega as imagens
macaco_parado = cv2.imread("macaco-parado.jpg")
macaco_dedo = cv2.imread("macaco-dedo-levantado.jpg")
macaco_pensando = cv2.imread("macaco-pensando.jpg")

# Proteção + aviso no console
if macaco_pensando is None:
    print("[AVISO] macaco_pensando.jpg não encontrado ou falhou ao carregar! Usando macaco_parado como fallback.")
    macaco_pensando = macaco_parado
else:
    print("[OK] macaco_pensando.jpg carregado com sucesso!")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Macaco", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 800, 600)
cv2.resizeWindow("Macaco", 800, 600)
cv2.moveWindow("Macaco", 100, 100)
cv2.moveWindow("Camera", 910, 100)
cv2.imshow("Macaco", macaco_parado)

cap = cv2.VideoCapture(0)
frames_sem_mao = 0
THRESHOLD = 5

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        indicador_levantado = False
        overlay = frame.copy()

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                lm = hand_landmarks
                points = []
                for landmark in lm:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    points.append((x, y))

                # Desenha linhas e bolinhas neon
                for connection in HAND_CONNECTIONS:
                    start = points[connection[0]]
                    end = points[connection[1]]
                    cv2.line(overlay, start, end, (255, 255, 255), 4)

                for (x, y) in points:
                    cv2.circle(overlay, (x, y), 10, (255, 255, 255), -1)
                    cv2.circle(overlay, (x, y), 7, (0, 0, 255), -1)

                # Condição rigorosa pro indicador levantado
                index_tip = lm[8]
                index_mcp = lm[5]
                middle_tip = lm[12]
                ring_tip = lm[16]
                pinky_tip = lm[20]

                if (index_tip.y < index_mcp.y - 0.15 and
                    index_tip.y < middle_tip.y + 0.1 and
                    index_tip.y < ring_tip.y + 0.1 and
                    index_tip.y < pinky_tip.y + 0.1):
                    indicador_levantado = True

        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # LÓGICA DO MACACO (com print pra debug)
        if indicador_levantado:
            frames_sem_mao = 0
            cv2.imshow("Macaco", macaco_dedo)
        elif result.hand_landmarks:
            frames_sem_mao = 0
            cv2.imshow("Macaco", macaco_pensando)
        else:
            frames_sem_mao += 1
            if frames_sem_mao > THRESHOLD:
                cv2.imshow("Macaco", macaco_parado)
        cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()