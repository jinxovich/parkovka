import cv2
from ultralytics import YOLO
import time
import torch

VIDEO_SOURCE = 'test1.mp4'  
OUTPUT_FILE = 'final_result.mp4'

MODEL_SIGNS = 'models/traffic_signs_v1.pt'
MODEL_ROAD = 'models/potholes_v1.pt'

CONF_THRESHOLD = 0.45 

def main():
    print(f"Загрузка моделей на {torch.cuda.get_device_name(0)}...")
    
    # Загружаем обе модели
    model_signs = YOLO(MODEL_SIGNS)
    model_road = YOLO(MODEL_ROAD)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌ Не могу открыть видео!")
        return

    # Параметры видео для сохранения
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Кодек для сохранения (mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (width, height))

    print("Обработка кадров...")
    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        
        # 1. Детектим ЗНАКИ
        # stream=True ускоряет процесс, так как не накапливает результаты в RAM
        results_signs = model_signs.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=0)
        
        # 2. Детектим ЯМЫ, ЛЮДЕЙ, МАШИНЫ
        results_road = model_road.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=0)

        # Мы используем встроенный плоттер YOLO, это самый быстрый способ
        
        annotated_frame = results_road[0].plot()
        
        r_signs = results_signs[0]
        if len(r_signs.boxes) > 0:
            r_signs.orig_img = annotated_frame
            annotated_frame = r_signs.plot(img=annotated_frame)

        cv2.imshow('Parkovka AI System', annotated_frame)
        
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nГотово! Обработано {frame_count} кадров за {total_time:.1f} сек.")
    print(f"⚡ Средний FPS: {frame_count / total_time:.1f}")
    print(f"💾 Результат сохранен в {OUTPUT_FILE}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()