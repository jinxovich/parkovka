from ultralytics import YOLO
import torch

def main():
    print(f"Обучение. GPU: {torch.cuda.get_device_name(0)}")
    
    model = YOLO('yolo11s.pt')  

    model.train(
        data='datasets/potholes/data.yaml', 
        
        epochs=30,             
        imgsz=640,
        batch=24,              
        workers=8,             
        cache=False,         
        
        device=0,              
        name='archive',   # <--- Имя папки результата
        optimizer='auto',      
        patience=10,           
        save=True,
        plots=True             
    )
    
    print("✅ Готово! Веса лежат в runs/detect/potholes_run/weights/best.pt")

if __name__ == '__main__':
    main()