import cv2
import numpy as np

# ===== CAMERA =====
cap = cv2.VideoCapture(0)
macaco_parado = cv2.imread("macaco-parado.jpg")
macaco_dedo = cv2.imread("macaco-dedo-levantado.jpg")

# ===== JANELAS =====
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Macaco", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Camera", 800, 600)
cv2.resizeWindow("Macaco", 300, 300)

cv2.imshow("Macaco", macaco_parado)

frames_sem_mao = 0
THRESHOLD = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Espelhar
    frame = cv2.flip(frame, 1)

    altura, largura, _ = frame.shape

    # ROI (lado direito da tela)
    x1 = largura * 2 // 3
    y1 = altura // 4
    x2 = largura
    y2 = altura * 3 // 4

    roi = frame[y1:y2, x1:x2]

    # Converter pra HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Range de pele (bem tolerante)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Quantidade de pixels brancos
    pixels_brancos = cv2.countNonZero(mask)
    area_total = (x2 - x1) * (y2 - y1)
    porcentagem = (pixels_brancos / area_total) * 100

    mao_detectada = porcentagem > 3.0

    if mao_detectada:
        frames_sem_mao = 0
        cv2.imshow("Macaco", macaco_dedo)
        cor = (0, 255, 0)  # verde
    else:
        frames_sem_mao += 1
        if frames_sem_mao > THRESHOLD:
            cv2.imshow("Macaco", macaco_parado)
        cor = (0, 0, 255)  # vermelho

    # Desenhar ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 3)

    cv2.imshow("Camera", frame)

    # ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
