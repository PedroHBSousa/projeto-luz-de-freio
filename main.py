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

# Buffer para estabilidade de detecção
detection_buffer = 0
buffer_threshold = 5

# Função para emitir som de alerta em loop
def play_alert_sound():
    global alert_playing
    while alert_playing:
        playsound(alert_sound_path)
        time.sleep(0.5)

# Função aprimorada para detectar luzes de freio
def detect_red_light(frame):
    global detection_buffer

    # Converter a imagem para o espaço de cor HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Ajuste dos limites para detectar uma gama maior de vermelho
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Criar máscara para vermelho
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = cv2.add(mask1, mask2)

    # Aumentar área de interesse (ROI)
    height, width = frame.shape[:2]
    roi = red_mask[height // 3:, :]  # Aumentar para os 2/3 inferiores da imagem

    # Calcular a média do brilho (V) da região de interesse
    roi_hsv = hsv_frame[height // 3:, :]  # Extrair a ROI do frame HSV
    brightness_avg = np.mean(roi_hsv[:, :, 2])  # Média do canal de brilho (V)

    # Ajustar limiar de brilho baseado no brilho médio da ROI
    dynamic_threshold = brightness_avg + 50  # Exemplo: valor fixo acima da média (ajustar conforme necessário)
    brightness_mask = roi_hsv[:, :, 2] > dynamic_threshold  # Máscara de brilho ajustada dinamicamente

    # Combine a máscara de cor vermelha com a de brilho
    combined_mask = cv2.bitwise_and(roi, roi, mask=brightness_mask.astype(np.uint8))

    # Encontrar contornos na máscara combinada
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Variável para indicar se luz vermelha foi detectada
    light_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Ajuste do tamanho mínimo do contorno
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Verificar se o contorno é semelhante ao de uma luz de freio
            if 0.4 < aspect_ratio < 2.5:
                cv2.rectangle(frame, (x, y + height // 3), (x + w, y + h + height // 3), (0, 255, 0), 2)
                light_detected = True

    # Controle de buffer para estabilidade
    if light_detected:
        detection_buffer += 1
    else:
        detection_buffer = max(0, detection_buffer - 1)

    # Confirmar detecção se buffer estiver acima do limiar
    return detection_buffer > buffer_threshold


# Função para listar dispositivos de câmera
def list_cameras(max_cameras=5):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(index)
            cap.release()
    return available_cameras

def main():
    global alert_playing

    cameras = list_cameras()
    if not cameras:
        print("Nenhuma câmera encontrada.")
        return

    print("Câmeras disponíveis:")
    for i, cam in enumerate(cameras):
        print(f"{i}: Câmera {cam}")

    camera_choice = int(input("Escolha o número da câmera que deseja usar: "))
    camera_index = cameras[camera_choice]

    cap = cv2.VideoCapture(camera_index)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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

    cap.release()
    cv2.destroyAllWindows()
    alert_playing = False

if __name__ == "__main__":
    main()
