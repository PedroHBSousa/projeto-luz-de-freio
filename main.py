import cv2
import numpy as np
from playsound import playsound
import threading
import os
import time
import logging
from tkinter import Tk, filedialog

# Configuração do logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Caminho do som de alerta
ALERT_SOUND_PATH = os.path.join("sounds", "alert.mp3")

# Variáveis globais
alert_playing = False
sound_thread_active = False  # Variável para controlar execução única do som
detection_buffer = 0
paused = False  # Variável para controlar a pausa
BUFFER_THRESHOLD = 5
MIN_CONTOUR_AREA = 500

# Função para emitir som de alerta
def play_alert_sound():
    """Reproduz som de alerta apenas uma vez por ciclo de detecção."""
    global sound_thread_active
    sound_thread_active = True
    try:
        playsound(ALERT_SOUND_PATH)
    except Exception as e:
        logging.error(f"Erro ao tocar o som de alerta: {e}")
    finally:
        sound_thread_active = False

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

    # Focar na região de interesse (inferior e central do frame)
    height, width = frame.shape[:2]
    roi = red_mask[height // 2:, width // 3: 2 * width // 3]  # Região central e inferior reduzida

    # Aplicar operações morfológicas para reduzir ruídos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    # Ajustar brilho dinamicamente
    roi_hsv = hsv_frame[height // 2:, width // 3: 2 * width // 3]  # Região correspondente para HSV
    brightness_avg = np.mean(roi_hsv[:, :, 2])
    brightness_threshold = brightness_avg + 50
    brightness_mask = roi_hsv[:, :, 2] > brightness_threshold
    combined_mask = cv2.bitwise_and(roi, roi, mask=brightness_mask.astype(np.uint8))

    # Encontrar contornos
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    light_detected = False

    # Lista para armazenar as luzes detectadas
    detected_lights = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.4 < aspect_ratio < 2.5:  # Aspecto semelhante a uma luz de freio
                # Calcular a distância do centro do ROI
                center_x = x + w // 2
                center_y = y + h // 2
                distance_to_center = abs(center_x - (width // 2)) + abs(center_y - (height // 2))
                detected_lights.append((x, y, w, h, distance_to_center))

    # Ordenar as luzes detectadas pela proximidade do centro
    detected_lights = sorted(detected_lights, key=lambda light: light[4])[:2]  # Selecionar as 2 mais próximas

    # Desenhar retângulos nas luzes detectadas
    for x, y, w, h, _ in detected_lights:
        cv2.rectangle(frame, (x + width // 3, y + height // 2), (x + w + width // 3, y + h + height // 2), (0, 255, 0), 2)
        light_detected = True

    # Atualizar buffer para suavizar detecção
    detection_buffer = min(BUFFER_THRESHOLD, detection_buffer + 1) if light_detected else max(0, detection_buffer - 1)

    return detection_buffer >= BUFFER_THRESHOLD

# Função principal
def main():
    """Executa o detector de luz vermelha com um arquivo de vídeo."""
    global alert_playing, paused, sound_thread_active

    # Janela de seleção de arquivo
    Tk().withdraw()  # Esconde a janela principal do Tkinter
    video_path = filedialog.askopenfilename(title="Selecione o arquivo de vídeo", filetypes=[("Arquivos de vídeo", "*.mp4;*.avi;*.mov;*.mkv;*.webm")])

    if not video_path:
        logging.error("Nenhum arquivo selecionado.")
        return

    if not os.path.exists(video_path):
        logging.error("Arquivo de vídeo não encontrado.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error("Não foi possível abrir o arquivo de vídeo.")
        return

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logging.info("Fim do vídeo ou falha ao ler frame.")
                    break

                # Detectar luz vermelha
                red_light_detected = detect_red_light(frame)

                # Gerenciar som de alerta
                if red_light_detected and not alert_playing:
                    alert_playing = True
                    if not sound_thread_active:
                        threading.Thread(target=play_alert_sound, daemon=True).start()
                elif not red_light_detected:
                    alert_playing = False

                # Exibir imagem
                cv2.imshow("Red Light Detector", frame)

            # Capturar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Sair
                break
            elif key == ord('p'):  # Pausar/retomar
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()
        alert_playing = False
        logging.info("Recursos liberados. Encerrando.")

if __name__ == "__main__":
    main()
