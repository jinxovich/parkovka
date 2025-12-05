from ultralytics import YOLO
import torch

def main():
    if not torch.cuda.is_available():
        print("ОШИБКА: PyTorch работает на CPU")
        return
    
    print(f"Видеокарта: {torch.cuda.get_device_name(0)}")
    
    # 1. Инициализация модели
    model = YOLO('yolo11s.pt')  

    # 2. Запуск обучения
    results = model.train(
        data='datasets/traffic_signs/data.yaml', 
        
        epochs=30,             
        
        imgsz=640,             
        
        batch=24,              
        
        workers=8,             
        
        cache=False,            
        
        device=0,              
        name='rtsd_ssd_run',   # Новое имя папки
        optimizer='auto',      
        patience=10,           
        save=True,             
        exist_ok=True,         
        amp=True,              
        plots=True             
    )
    
    # 3. Финал
    print("Обучение завершено. Идет валидация...")
    metrics = model.val()
    print(f"Итоговый mAP50: {metrics.box.map50}")


if __name__ == '__main__':
    main()