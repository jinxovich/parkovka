import os

def main():
    # 1. Путь к новой папке с ямами
    project_root = os.getcwd()
    dataset_dir = os.path.join(project_root, 'datasets', 'potholes')
    yaml_path = os.path.join(dataset_dir, 'data.yaml')

    print(f"🔧 Настраиваем конфиг для ЯМ: {yaml_path}")

    if not os.path.exists(yaml_path):
        print("❌ Ошибка: Не нашел data.yaml в datasets/potholes/")
        print("Убедись, что ты переместил файлы туда.")
        return

    # 2. Читаем список классов из файла
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Выдираем names и nc
    if "names:" in content:
        names_block = content[content.find("names:"):]
        # Пробуем найти nc (количество классов)
        if "nc:" in content:
            import re
            nc_match = re.search(r"nc:\s*(\d+)", content)
            nc_line = f"nc: {nc_match.group(1)}" if nc_match else "nc: 1" # Дефолт
        else:
            # Если nc нет, посчитаем строки в names (грубо, но сработает)
            nc_line = "nc: 3" # Обычно там Pothole, Crack, etc.
    else:
        print("❌ Ошибка: Странный формат data.yaml")
        return

    # 3. Перезаписываем с абсолютными путями
    new_yaml = f"""
path: {dataset_dir}
train: train/images
val: valid/images
test: test/images

{nc_line}
{names_block}
"""
    
    with open(yaml_path, 'w') as f:
        f.write(new_yaml)

    print("✅ Готово! Конфиг исправлен.")

if __name__ == '__main__':
    main()