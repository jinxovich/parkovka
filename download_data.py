from roboflow import Roboflow
import os
import shutil
import yaml

# --- КОНФИГУРАЦИЯ ---
# Вставь сюда свой НОВЫЙ ключ
KEY = "POyjJe1KQO4dTPOutaU1" 
TARGET_DIR = "datasets/vehicles"
ROBOFLOW_WORKSPACE = "roboflow"
ROBOFLOW_PROJECT = "self-driving-car"
ROBOFLOW_VERSION = 3  # Стабильная версия с машинами, пешеходами, светофорами

def fix_yaml_paths(dataset_path):
    """
    Исправляет пути в data.yaml, чтобы они были абсолютными 
    для ТВОЕГО компьютера, а не путями из облака.
    """
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print("❌ data.yaml не найден!")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Делаем пути абсолютными от текущей папки
    abs_path = os.path.abspath(dataset_path)
    
    data['path'] = abs_path
    data['train'] = "train/images"
    data['val'] = "valid/images"
    data['test'] = "test/images"
    
    # Убираем лишние ключи, если они есть (Roboflow иногда добавляет мусор)
    if 'names' in data and isinstance(data['names'], list):
        # Превращаем список в словарь индексов, если нужно, или оставляем как есть
        # YOLOv8 понимает и списки, и словари.
        pass

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"🔧 Config data.yaml исправлен: пути настроены на {abs_path}")

def main():
    rf = Roboflow(api_key=KEY)
    print(">>> Начинаю скачивание датасета Self Driving Car...")

    try:
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(ROBOFLOW_VERSION)
        
        # Скачиваем во временную папку
        dataset = version.download("yolov8")
        downloaded_folder = dataset.location

        # Логика перемещения в datasets/vehicles
        if os.path.exists(TARGET_DIR):
            print(f">>> Чищу старую папку {TARGET_DIR}...")
            shutil.rmtree(TARGET_DIR)
        
        # Переименовываем/Перемещаем скачанную папку
        print(f">>> Перемещаю файлы в {TARGET_DIR}...")
        # Roboflow качает в папку с названием проекта, нам надо переименовать её
        # dataset.location хранит путь куда скачалось.
        shutil.move(downloaded_folder, TARGET_DIR)
        
        # Фиксим пути в YAML
        fix_yaml_paths(TARGET_DIR)

        print(f"\n✅ УСПЕХ! Датасет готов к работе в: {TARGET_DIR}")
        print("Классы: Biker, Car, Pedestrian, TrafficLight, Truck")

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("Проверь API KEY и подключение к интернету.")

if __name__ == "__main__":
    main()