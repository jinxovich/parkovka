from ultralytics import YOLO
import torch
import os

# Магия для лечения фрагментации памяти (как просил PyTorch в ошибке)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    if not torch.cuda.is_available():
        print("ОШИБКА: PyTorch работает на CPU!")
        return
    
    print(f"Видеокарта: {torch.cuda.get_device_name(0)}")
    
    # Грузим Medium модель
    model = YOLO('yolo11m.pt')  

    print(">>> Обучаем только классы: 4(crack), 8(manhole), 11(speed_bump)")

    # Запуск обучения
    results = model.train(
        data='datasets/road_surface/data.yaml', 
        
        epochs=100,            
        imgsz=640,             
  
        batch=16,              
        
        workers=8,             
        cache=True,            
        device=0,              
        name='road_surface_v1',
        optimizer='auto',      
        patience=15,           
        save=True,             
        exist_ok=True,         
        amp=True,              
        plots=True,
        
        # Фильтр классов (твои индексы)
        classes=[4, 8, 11]     
    )
    
    print("Обучение завершено. Идет валидация...")
    metrics = model.val()
    print(f"Итоговый mAP50: {metrics.box.map50}")


if __name__ == '__main__':
    main()