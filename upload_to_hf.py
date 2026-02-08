import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

# Загружаем ключи из .env для отправки в Secrets
load_dotenv()

# --- ВАШИ ДАННЫЕ ---
# Токен с правами WRITE (можно взять на сайте HF: Settings -> Access Tokens)
HF_TOKEN = "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" 
# Ваше имя пользователя на HF и название, которое хотите дать Space
SPACE_ID = "YakubQ2/neuro-ascii-api"  # Например: "Orb/NeuroASCII"

# Загружаемая папка (текущая)
LOCAL_DIR = Path(__file__).parent.resolve()

def upload_project_to_space():
    if HF_TOKEN == "ВСТАВЬТЕ_СЮДА_ВАШ_ТОКЕН":
        print("❌ Ошибка: Вы не вставили свой HF_TOKEN в скрипт!")
        return
        
    print(f"Инициализация загрузки в Space: [ {SPACE_ID} ]...")
    api = HfApi(token=HF_TOKEN)
    
    # 1. Создаем Space (это безопасно, если он уже есть - 그냥 продолжит)
    try:
        print("Создание/Проверка Space (SDK: docker)...")
        create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="docker",
            token=HF_TOKEN,
            exist_ok=True
        )
    except Exception as e:
        print(f"⚠️ Space может уже существовать или произошла ошибка: {e}")

    # 2. Добавляем Секреты (Шаг 4)
    print("Настройка секретов (UNSPLASH и PEXELS)...")
    keys_to_add = ["UNSPLASH_ACCESS_KEY", "PEXELS_API_KEY"]
    for key in keys_to_add:
        val = os.environ.get(key)
        if val:
            try:
                api.add_space_secret(repo_id=SPACE_ID, key=key, value=val)
                print(f"  [OK] Секрет {key} успешно добавлен в Space!")
            except Exception as e:
                print(f"  [ERROR] Ошибка при добавлении секрета {key}: {e}")
        else:
            print(f"  [INFO] {key} не найден в вашем .env, пропускаем.")

    # 3. Загрузка всех файлов
    print(f"Загрузка файлов из папки: {LOCAL_DIR}...")
    # Игнорируем ненужные/приватные файлы
    ignore_patterns = [
        ".env",             # Секреты заливаем только как Secrets!
        "__pycache__/", 
        "*.pyc",
        ".git/",
        ".gitignore",
        "data/*.tar.gz",           # 5GB датасеты не нужны для работы
        "data/coco/",              # сырые картинки
        "data/food-101/",          # сырые картинки
        "data/cifar-100-python/",  # сырые картинки
        "storage/weights/pretrain*.pt",  # промежуточные веса
        "storage/weights/v2_epoch*.pt",  # эпохи
        "storage/weights/neuroascii_best.pt", # старые веса
        "storage/weights/neuroascii_latest.pt", # старые веса
        "storage/weights/neuroascii_v2_best.pt", # дубликаты
        "tests/",
        "training/",
        "*.bat",
        upload_to_hf_script_name # Сам этот скрипт можно не грузить
    ]
    
    try:
        url = upload_folder(
            folder_path=str(LOCAL_DIR),
            repo_id=SPACE_ID,
            repo_type="space",
            token=HF_TOKEN,
            ignore_patterns=ignore_patterns,
            commit_message="Upload NeuroASCII project via API"
        )
        print("\nГОТОВО! Проект успешно загружен!")
        print(f"Ваш Space доступен по ссылке: {url}")
        print("Теперь Hugging Face начнет сборку Docker контейнера. Зайдите на сайт и проверьте вкладку 'App'!")
    except Exception as e:
        print(f"Критическая ошибка при загрузке файлов: {e}")

if __name__ == "__main__":
    upload_to_hf_script_name = Path(__file__).name
    upload_project_to_space()
