# страница для задания Саши

import os
import json
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import vgg16, resnet18
import time
from PIL import Image
import matplotlib.pyplot as plt

st.title('Предсказание случайной картинки какой-то моделью :)')

# установка устройства и загрузка модели
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")


# Получаем директорию текущего файла (random_images.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Переходим на уровень выше к корневой директории проекта
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Конструируем путь к файлу весов для VGG16
weights_path_VGG = os.path.join(project_root, 'models', 'sasha', 'model_VGG_state_dict.pth')
# Конструируем путь к файлу весов для ResNet18
weights_path_res = os.path.join(project_root, 'models', 'sasha', 'model_res_state_dict.pth')

# загрузка модели VGG16
@st.cache_resource
def load_model_VGG():
    # Создаем архитектуру модели без предобученных весов
    model = vgg16(weights=None)

    # Загружаем сохраненные веса
    model.load_state_dict(torch.load(weights_path_VGG, map_location=DEVICE))

    # Перемещаем модель на выбранное устройство
    model.to(DEVICE)
    model.eval()
    return model

model_VGG = load_model_VGG()

# загрузка модели ResNet18
@st.cache_resource
def load_model_res():
    # Создаем архитектуру модели без предобученных весов
    model = resnet18(weights=None)

    # Загружаем сохраненные веса
    model.load_state_dict(torch.load(weights_path_res, map_location=DEVICE))

    # Перемещаем модель на выбранное устройство
    model.to(DEVICE)
    model.eval()
    return model

model_res = load_model_res()

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
    st.write('Вы хотите определить, что на картинке ниже:')
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    # Кнопка для запуска предсказания
    if st.button('Посмотреть предсказание!'):
        # Преобразование изображения
        image_tensor = transform(image).to(DEVICE).unsqueeze(0)
        
        ### модель VGG 16
        # Замер времени
        start_time_VGG = time.time()
        with torch.inference_mode():
            outputs_VGG = model_VGG(image_tensor)
        end_time_VGG = time.time()
        inference_time_VGG = end_time_VGG - start_time_VGG
        # st.write(f"**Время выполнения предсказания:** {inference_time:.6f} секунд")
        
        # Обработка результатов
        probabilities_VGG = F.softmax(outputs_VGG, dim=1)
        top5_prob_VGG, top5_catid_VGG = torch.topk(probabilities_VGG, 5)
        top5_prob_VGG = top5_prob_VGG.cpu().squeeze().tolist()
        top5_catid_VGG = top5_catid_VGG.cpu().squeeze().tolist()
        top5_classes_VGG = [decode(catid) for catid in top5_catid_VGG]
        
        st.write('Предсказания моделью VGG16')
        # Создаем две колонки
        col1_VGG, col2_VGG = st.columns([1, 2])

        with col1_VGG:
            # Отображение обработанного изображения с предсказанной меткой
            fig, ax = plt.subplots()
            # Обратное преобразование нормализации для корректного отображения
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
            image_unorm = image_tensor.cpu().squeeze() * std + mean
            image_unorm = torch.clamp(image_unorm, 0, 1)
            ax.imshow(torch.permute(image_unorm, (1, 2, 0)))
            ax.axis('off')
            ax.set_title(top5_classes_VGG[0])
            st.pyplot(fig)

        with col2_VGG:
            st.write(f"**Время выполнения предсказания:** {inference_time_VGG:.6f} секунд")
            st.write("**Топ-5 предсказаний:**")
            for i in range(len(top5_classes_VGG)):
                st.write(f"{i+1}. {top5_classes_VGG[i]}: {top5_prob_VGG[i]*100:.2f}%")
                
        ### модель ResNet18
        # Замер времени
        start_time_res = time.time()
        with torch.inference_mode():
            outputs_res = model_res(image_tensor)
        end_time_res = time.time()
        inference_time_res = end_time_res - start_time_res
        # st.write(f"**Время выполнения предсказания:** {inference_time:.6f} секунд")
        
        # Обработка результатов
        probabilities_res = F.softmax(outputs_res, dim=1)
        top5_prob_res, top5_catid_res = torch.topk(probabilities_res, 5)
        top5_prob_res = top5_prob_res.cpu().squeeze().tolist()
        top5_catid_res = top5_catid_res.cpu().squeeze().tolist()
        top5_classes_res = [decode(catid) for catid in top5_catid_res]
        
        st.write('Предсказания моделью ResNet18')
        # Создаем две колонки
        col1_res, col2_res = st.columns([1, 2])

        with col1_res:
            # Отображение обработанного изображения с предсказанной меткой
            fig, ax = plt.subplots()
            # Обратное преобразование нормализации для корректного отображения
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
            image_unorm = image_tensor.cpu().squeeze() * std + mean
            image_unorm = torch.clamp(image_unorm, 0, 1)
            ax.imshow(torch.permute(image_unorm, (1, 2, 0)))
            ax.axis('off')
            ax.set_title(top5_classes_res[0])
            st.pyplot(fig)

        with col2_res:
            st.write(f"**Время выполнения предсказания:** {inference_time_res:.6f} секунд")
            st.write("**Топ-5 предсказаний:**")
            for i in range(len(top5_classes_res)):
                st.write(f"{i+1}. {top5_classes_res[i]}: {top5_prob_res[i]*100:.2f}%")
        
        # # Вывод топ-5 предсказаний
        # st.write("**Топ-5 предсказаний:**")
        # for i in range(len(top5_classes)):
        #     st.write(f"{i+1}. {top5_classes[i]}: {top5_prob[i]*100:.2f}%")

