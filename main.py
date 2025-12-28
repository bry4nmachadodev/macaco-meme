import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

macaco_parado = cv2.imread("macaco-parado.jpg")
macaco_dedo = cv2.imread("macaco-dedo-levantado.jpg")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,   # Baixei um pouco pra detectar mais fácil
    min_hand_presence_confidence=0.5,    # Mais sensível
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
frames_sem_indicador = 0
THRESHOLD = 5

# Conexões da mão
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

        # Verifica se pelo menos uma mão foi detectada (mesmo sem landmarks completos)
        if result.hand_landmarks or result.handedness:  # handedness existe mesmo com landmarks parciais
            mao_detectada = True
        else:
            mao_detectada = False

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                lm = hand_landmarks
                points = []
                for landmark in lm:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    points.append((x, y))

                # Linhas brancas mais finas
                for connection in HAND_CONNECTIONS:
                    start = points[connection[0]]
                    end = points[connection[1]]
                    cv2.line(overlay, start, end, (255, 255, 255), 4)  # Era 8, agora 4 (mais fina)

                # Bolinhas menores e mais elegantes
                for (x, y) in points:
                    cv2.circle(overlay, (x, y), 10, (255, 255, 255), -1)  # Glow branco menor
                    cv2.circle(overlay, (x, y), 7, (0, 0, 255), -1)       # Vermelho menor

                # Condição do indicador (um pouco mais flexível)
                index_tip = lm[8]
                index_mcp = lm[5]
                middle_tip = lm[12]
                ring_tip = lm[16]
                pinky_tip = lm[20]

                if (index_tip.y < index_mcp.y - 0.12 and   # Um pouco menos rigoroso
                    index_tip.y < middle_tip.y - 0.05 and  # Tolerância pros outros dedos
                    index_tip.y < ring_tip.y - 0.05 and
                    index_tip.y < pinky_tip.y - 0.05):
                    indicador_levantado = True

        # Mistura o overlay com o frame original (glow suave)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Atualiza o macaco
        if indicador_levantado:
            frames_sem_indicador = 0
            cv2.imshow("Macaco", macaco_dedo)
        else:
            frames_sem_indicador += 1
            if frames_sem_indicador > THRESHOLD:
                cv2.imshow("Macaco", macaco_parado)

        cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()