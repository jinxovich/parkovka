import json
import os
import shutil
from tqdm import tqdm
from PIL import Image

BASE_DIR = 'archive'
# Путь к папке с исодниками
IMAGES_SOURCE_DIR = os.path.join(BASE_DIR, 'rtsd-frames', 'rtsd-frames') 

TRAIN_JSON = os.path.join(BASE_DIR, 'train_anno.json')
VAL_JSON = os.path.join(BASE_DIR, 'val_anno.json')

OUTPUT_DIR = 'datasets/traffic_signs'

def convert_coco_to_yolo(json_file, subset_name, categories_map=None):
    print(f"\nЗагрузка {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 1. Создаем карту категорий 
    # COCO формат: "categories": [{"id": 1, "name": "sign_a"}, ...]
    if categories_map is None:
        categories_map = {}
        # Сортируем категории по ID, чтобы порядок был детерминированным
        sorted_cats = sorted(data['categories'], key=lambda x: x['id'])
        
        # Создаем mapping: real_cat_id -> yolo_id (0, 1, 2...)
        # охраним список имен для data.yaml
        yolo_id = 0
        real_to_yolo = {}
        yolo_names = {}
        
        for cat in sorted_cats:
            real_id = cat['id']
            name = cat['name']
            real_to_yolo[real_id] = yolo_id
            yolo_names[yolo_id] = name
            yolo_id += 1
        
        categories_map = {'real_to_yolo': real_to_yolo, 'names': yolo_names}
    else:
        real_to_yolo = categories_map['real_to_yolo']

    # 2. Индексируем изображения: image_id -> file_name
    images_info = {}
    for img in data['images']:
        images_info[img['id']] = img

    # 3. Группируем аннотации по image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # 4. Подготовка папок
    img_out_dir = os.path.join(OUTPUT_DIR, 'images', subset_name)
    lbl_out_dir = os.path.join(OUTPUT_DIR, 'labels', subset_name)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    print(f"Конвертация {subset_name}...")
    
    missing_files = 0
    
    for img_id, img_data in tqdm(images_info.items()):

        filename_raw = img_data['file_name']
        filename = os.path.basename(filename_raw)
        
        src_path = os.path.join(IMAGES_SOURCE_DIR, filename)
        
        if not os.path.exists(src_path):
            missing_files += 1
            continue

        # символическая ссылка на изображение
        dst_path = os.path.join(img_out_dir, filename)
        os.symlink(os.path.abspath(src_path), dst_path)

        # Размеры изображения
        img_w = img_data['width']
        img_h = img_data['height']

        # Создаем .txt
        txt_name = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(lbl_out_dir, txt_name)

        with open(txt_path, 'w') as f_txt:
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    cat_id = ann['category_id']
                    
                    if cat_id not in real_to_yolo:
                        continue 
                    
                    cls_id = real_to_yolo[cat_id]
                    
                    # COCO bbox: [x_min, y_min, width, height]
                    bbox = ann['bbox']
                    x_min, y_min, w_box, h_box = bbox[0], bbox[1], bbox[2], bbox[3]

                    # Перевод в YOLO: center_x, center_y, w, h (normalized)
                    x_center = (x_min + w_box / 2) / img_w
                    y_center = (y_min + h_box / 2) / img_h
                    w_norm = w_box / img_w
                    h_norm = h_box / img_h
                    
                    # Ограничение 0-1
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))

                    f_txt.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    if missing_files > 0:
        print(f"Не найдено файлов изображений: {missing_files} (возможно, битые пути в JSON)")
        
    return categories_map

def main():
    # 1. Обработка TRAIN
    # Получим карту классов из трейна
    cat_map = convert_coco_to_yolo(TRAIN_JSON, 'train')

    # 2. Обработка VAL
    if os.path.exists(VAL_JSON):
        convert_coco_to_yolo(VAL_JSON, 'val', categories_map=cat_map)
    
    # 3. Создаем data.yaml
    print("Создание data.yaml")
    
    names_dict = cat_map['names']
    # Сортируем имена по ID (0, 1, 2...)
    sorted_names = [names_dict[i] for i in range(len(names_dict))]
    
    yaml_content = f"""
path: ../{OUTPUT_DIR}
train: images/train
val: images/val

nc: {len(sorted_names)}
names: {sorted_names}
"""
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
        
    print(f"Готово")

if __name__ == '__main__':
    main()