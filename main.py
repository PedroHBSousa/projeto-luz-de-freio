import cv2
import numpy as np
from playsound import playsound
import threading
import os
import time

# Caminho do som de alerta
alert_sound_path = os.path.join("sounds", "alert.mp3")

# Variável para controle do som de alerta
alert_playing = False

# Função para emitir som de alerta em loop
def play_alert_sound():
    global alert_playing
    while alert_playing:  # Loop enquanto alert_playing for True
        playsound(alert_sound_path)
        time.sleep(0.5)  # Pequeno intervalo para evitar sobreposição excessiva

# Função para detectar cor vermelha
def detect_red_light(frame):
    # Converter a imagem para o espaço de cor HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir limites para a cor vermelha
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Filtrar a cor vermelha
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.add(mask1, mask2)

    # Processamento para detecção de área vermelha
    red_output = cv2.bitwise_and(frame, frame, mask=red_mask)
    gray = cv2.cvtColor(red_output, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filtrar contornos pequenos
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return True  # Luz vermelha detectada

    return False

def main():
    global alert_playing
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar luz vermelha
        red_light_detected = detect_red_light(frame)

        # Emitir som de alerta se luz vermelha detectada
        if red_light_detected and not alert_playing:
            alert_playing = True
            threading.Thread(target=play_alert_sound, daemon=True).start()
        elif not red_light_detected and alert_playing:
            alert_playing = False

        # Exibir a imagem capturada
        cv2.imshow("Red Light Detector", frame)

        # Encerrar com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    alert_playing = False  # Parar o som quando o programa for encerrado

if __name__ == "__main__":
    main()
