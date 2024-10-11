# страница для задания Саши
# pages/3_Random_images.py

import os
import json
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import vgg16, VGG16_Weights
import requests
import time
from PIL import Image


# Интерфейс Streamlit
def main():
    st.title("Классификация изображений с помощью VGG16")

# установка устройства и загрузка модели
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")


# Получаем директорию текущего файла (random_images.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Переходим на уровень выше к корневой директории проекта
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Конструируем путь к файлу весов
weights_path = os.path.join(project_root, 'models', 'sasha', 'model_VGG_state_dict.pth')

@st.cache_resource
def load_model():
    # Создаем архитектуру модели без предобученных весов
    model = vgg16(weights=None)

    # Загружаем сохраненные веса
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    # Перемещаем модель на выбранное устройство
    model.to(DEVICE)
    model.eval()
    return model

model_VGG = load_model()

# Загрузка меток классов
@st.cache_resource
def load_labels():
    labels_path = os.path.join(project_root, 'models', 'sasha', 'imagenet_class_index.json')
    # Загружаем метки из локального файла
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    decode = lambda x: labels[str(x)][1]
    return decode

decode = load_labels()

# Преобразования изображений
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    
    # Преобразование изображения
    image_tensor = transform(image).to(DEVICE).unsqueeze(0)
    
    # Замер времени
    start_time = time.time()
    with torch.inference_mode():
        outputs = model_VGG(image_tensor)
    end_time = time.time()
    inference_time = end_time - start_time
    st.write(f"**Время выполнения предсказания:** {inference_time:.6f} секунд")
    
    # Обработка результатов
    probabilities = F.softmax(outputs, dim=1)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5_prob = top5_prob.cpu().squeeze().tolist()
    top5_catid = top5_catid.cpu().squeeze().tolist()
    top5_classes = [decode(catid) for catid in top5_catid]
    
    # Вывод топ-5 предсказаний
    st.write("**Топ-5 предсказаний:**")
    for i in range(len(top5_classes)):
        st.write(f"{i+1}. {top5_classes[i]}: {top5_prob[i]*100:.2f}%")

