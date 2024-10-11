# страница для задания Дианы
import streamlit as st
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
from torchvision.models import resnet18, ResNet18_Weights

from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, ColorJitter, RandomCrop
from PIL import Image

import torchutils as tu
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Создание страницы Streamlit
st.title("Классификация кофейных зерен")

# Загрузка изображения пользователем
uploaded_file = st.file_uploader("Загрузите фотографию кофейного зерна", type=['jpg', 'jpeg', 'png'])

# КОД с моделью
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Создание экземпляра модели ResNet
model = torchvision.models.resnet18(pretrained=True)
# Заменяем последний слой для задачи классификации кофе
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4) # 4 класса кофе
# Замораживаем все параметры, кроме классификационного слоя
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

# Получаем директорию текущего файла (random_images.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Переходим на уровень выше к корневой директории проекта
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Конструируем путь к файлу весов
weights_path = os.path.join(project_root, 'models', 'diana', 'model_coffe.pth')

model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
model.eval()
model = model.to(DEVICE)

# Преобразования для изображения
preprocess = T.Compose([
                       T.ColorJitter(),
                       T.ToTensor(),
                       T.Resize((224, 224))
                       ])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Предсказание модели
    image = preprocess(image).unsqueeze(0)
    image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        output = model(image)

    # Обработка результата
    predicted_class = torch.argmax(output).item()
    class_name = ['Dark', 'Green', 'Light', 'Medium']

    st.write(f"Предсказанный класс: {class_name[predicted_class]}")