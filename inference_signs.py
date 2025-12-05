from ultralytics import YOLO
import os


VIDEO_SOURCE = 'test1.mp4'


MODEL_PATH = 'runs/detect/rtsd_ssd_run/weights/best.pt'

def main():

    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Не найдены веса модели {MODEL_PATH}")
        return

    print(f"Загрузка модели: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"апуск обработки видео: {VIDEO_SOURCE}")
    print("Нажми 'Q' в окне просмотра, чтобы прервать.")

    # Инференс
    results = model.predict(
        source=VIDEO_SOURCE,
        save=True,          # Сохранить видео с квадратиками
        show=True,          # Показать окно проигрывания
        conf=0.4,           # Порог уверенности (40%). Если много шума - ставь 0.5 или 0.6
        iou=0.5,            # Порог наложения (чтобы не было двойных рамок)
        device=0,           # Твоя RTX 3080
        imgsz=640,          # Размер, на котором учили
        line_width=2,       # Толщина рамок
        name='my_video_result' # Имя папки куда сохранится
    )

    print("\n✅ Готово! Результат сохранен в папке runs/detect/my_video_result")

if __name__ == '__main__':
    main()