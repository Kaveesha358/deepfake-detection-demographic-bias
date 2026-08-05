# Filter Images

from deepface import DeepFace
import os, shutil, random

src_real = "/kaggle/input/datasets/ayushmandatta1/deepdetect-2025/ddata/train/real"
src_fake = "/kaggle/input/datasets/ayushmandatta1/deepdetect-2025/ddata/train/fake"

def filter_asian(src_folder, out_folder, target_count):
    os.makedirs(out_folder, exist_ok=True)
    files = os.listdir(src_folder)
    random.shuffle(files)
    count = 0
    checked = 0
    for f in files:
        if count >= target_count:
            break
        checked += 1
        path = os.path.join(src_folder, f)
        try:
            result = DeepFace.analyze(path, actions=['race'], enforce_detection=False)
            dominant = result[0]['dominant_race']
            if dominant == 'asian':
                shutil.copy(path, out_folder)
                count += 1
        except:
            pass
        if checked % 20 == 0:
            print(f"Checked {checked}, found {count}")
    print(f"DONE {out_folder}: {count} images from {checked} checked")

filter_asian(src_real, "/kaggle/working/filtered_real", 500)
filter_asian(src_fake, "/kaggle/working/filtered_fake", 500)
