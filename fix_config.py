import os
import yaml

# Абсолютный путь к папке датасета
dataset_path = os.path.abspath("datasets/road_surface")
yaml_path = os.path.join(dataset_path, "data.yaml")

print(f"🔍 Проверяю папку: {dataset_path}")

# 1. Ищем правильные названия папок
dirs = os.listdir(dataset_path)
train_dir = "train" if "train" in dirs else None
val_dir = None
test_dir = None

# Робофлоу иногда называет valid, иногда val
if "valid" in dirs:
    val_dir = "valid"
elif "val" in dirs:
    val_dir = "val"

if "test" in dirs:
    test_dir = "test"

# Если папки train нет - беда
if not train_dir:
    print("❌ ОШИБКА: Не найдена папка train! Возможно, датасет скачался криво.")
    exit()

print(f"✅ Найдено: train='{train_dir}', val='{val_dir}', test='{test_dir}'")

# 2. Формируем правильный конфиг
new_config = {
    "path": dataset_path,
    "train": f"{train_dir}/images",
    "val": f"{val_dir}/images" if val_dir else f"{train_dir}/images", # Если нет val, валидируем на train (костыль, но сработает)
    "names": {
        0: "bike",
        1: "bus",
        2: "car",
        3: "cone",
        4: "crack",
        5: "face",
        6: "large_truck",
        7: "license_plate",
        8: "manhole",
        9: "person",
        10: "small_truck",
        11: "speed_bump"
    },
    "nc": 12
}

if test_dir:
    new_config["test"] = f"{test_dir}/images"

# 3. Перезаписываем data.yaml
with open(yaml_path, "w") as f:
    yaml.dump(new_config, f, sort_keys=False)

print(f"Файл {yaml_path} успешно исправлен.")