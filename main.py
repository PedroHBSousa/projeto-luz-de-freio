import cv2
import numpy as np
from playsound import playsound
import threading
import os
import time
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Caminho do som de alerta
ALERT_SOUND_PATH = os.path.join("sounds", "alert.mp3")

# Variáveis globais
alert_playing = False
detection_buffer = 0
BUFFER_THRESHOLD = 5
MIN_CONTOUR_AREA = 500

# Função para emitir som de alerta em loop
def play_alert_sound():
    """Reproduz som de alerta enquanto alert_playing estiver True."""
    global alert_playing
    try:
        while alert_playing:
            playsound(ALERT_SOUND_PATH)
            time.sleep(0.5)
    except Exception as e:
        logging.error(f"Erro ao tocar o som de alerta: {e}")

# Função aprimorada para detectar luzes de freio
def detect_red_light(frame):
    """Detecta luz vermelha no frame e usa buffer para estabilidade."""
    global detection_buffer

    # Converter a imagem para o espaço de cor HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Limites para detecção de vermelho
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Criar máscara para vermelho
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.add(mask1, mask2)

    # Focar na região de interesse (inferior do frame)
    height, width = frame.shape[:2]
    roi = red_mask[height // 3:, :]  # Região inferior do frame

    # Ajustar brilho dinamicamente
    roi_hsv = hsv_frame[height // 3:, :]
    brightness_avg = np.mean(roi_hsv[:, :, 2])
    brightness_threshold = brightness_avg + 50
    brightness_mask = roi_hsv[:, :, 2] > brightness_threshold
    combined_mask = cv2.bitwise_and(roi, roi, mask=brightness_mask.astype(np.uint8))

    # Encontrar contornos
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    light_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.4 < aspect_ratio < 2.5:  # Aspecto semelhante a uma luz de freio
                cv2.rectangle(frame, (x, y + height // 3), (x + w, y + h + height // 3), (0, 255, 0), 2)
                light_detected = True

    # Atualizar buffer para suavizar detecção
    detection_buffer = min(BUFFER_THRESHOLD, detection_buffer + 1) if light_detected else max(0, detection_buffer - 1)

    return detection_buffer >= BUFFER_THRESHOLD

# Função para listar dispositivos de câmera
def list_cameras(max_cameras=5):
    """Retorna uma lista de índices de câmeras disponíveis."""
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(index)
            cap.release()
    return available_cameras

# Função principal
def main():
    """Executa o detector de luz vermelha."""
    global alert_playing

    cameras = list_cameras()
    if not cameras:
        logging.error("Nenhuma câmera encontrada.")
        return

    logging.info("Câmeras disponíveis:")
    for i, cam in enumerate(cameras):
        logging.info(f"{i}: Câmera {cam}")

    try:
        camera_choice = int(input("Escolha o número da câmera que deseja usar: "))
        if camera_choice < 0 or camera_choice >= len(cameras):
            raise ValueError("Índice inválido.")
    except ValueError as e:
        logging.error("Entrada inválida. Encerrando.")
        return

    camera_index = cameras[camera_choice]
    cap = cv2.VideoCapture(camera_index)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Falha ao capturar frame. Encerrando.")
                break

            # Detectar luz vermelha
            red_light_detected = detect_red_light(frame)

            # Gerenciar som de alerta
            if red_light_detected and not alert_playing:
                alert_playing = True
                threading.Thread(target=play_alert_sound, daemon=True).start()
            elif not red_light_detected and alert_playing:
                alert_playing = False

            # Exibir imagem
            cv2.imshow("Red Light Detector", frame)

            # Tecla 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        alert_playing = False
        logging.info("Recursos liberados. Encerrando.")

if __name__ == "__main__":
    main()
