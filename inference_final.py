import cv2
from ultralytics import YOLO
import time
import torch
import os

VIDEO_SOURCE = 'test1.mp4'  
OUTPUT_FILE = 'final_result.mp4'

MODEL_SIGNS = 'models/traffic_signs_v1.pt'  
MODEL_ROAD = 'models/road_surface_v1.pt'    

# Разные пороги для разных задач
CONF_SIGNS = 0.50  
CONF_ROAD = 0.25   

def main():
    # Проверка путей
    if not os.path.exists(MODEL_ROAD):
        print(f"❌ ОШИБКА: Не найден файл {MODEL_ROAD}")
        print("Скопируй runs/detect/road_surface_v1/weights/best.pt в папку models/ и назови road_surface_v1.pt")
        return

    print(f"🚀 Загрузка моделей на {torch.cuda.get_device_name(0)}...")
    
    # Грузим модели
    try:
        model_signs = YOLO(MODEL_SIGNS)
        model_road = YOLO(MODEL_ROAD)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌ Не могу открыть видео!")
        return

    # Параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (width, height))

    print(f"Начало обработки: {width}x{height} @ {fps}FPS")
    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # --- 1. Детекция ДОРОГИ (Ямы, Люки, Лежачие) ---
        #  conf пониже, чтобы видеть трещины
        results_road = model_road.predict(frame, conf=CONF_ROAD, verbose=False, device=0)
        
        # --- 2. Детекция ЗНАКОВ ---
        #  conf повыше, чтобы не ловить мусор
        results_signs = model_signs.predict(frame, conf=CONF_SIGNS, verbose=False, device=0)

        # --- ОТРИСОВКА (Слоями) ---
        
        # Слой 1: Рисуем ямы на чистом кадре
        # plot() возвращает numpy массив (картинку)
        annotated_frame = results_road[0].plot(line_width=2) 
        
        # Слой 2: Рисуем знаки ПОВЕРХ результата с ямами
        # Аргумент img=annotated_frame заставляет рисовать на уже готовой картинке
        annotated_frame = results_signs[0].plot(img=annotated_frame, line_width=2)

        # Показ
        cv2.imshow('Parkovka AI: Road + Signs', annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    total_time = end_time - start_time
    
    print("-" * 30)
    print(f"Готово!")
    print(f"Кадров: {frame_count}")
    print(f"Время: {total_time:.1f} сек")
    print(f"Средний FPS: {frame_count / total_time:.1f}")
    print(f"Результат сохранен в: {OUTPUT_FILE}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()