import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# Загрузите сохранённое состояние модели
model_path = 'models/danila/model_50.pth'
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 30)  # 30 - количество классов в задаче
model.load_state_dict(torch.load(model_path))
model.eval()

# Функция для предсказания
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Интерфейс Streamlit
st.title('Классификация изображений с помощью ResNet50')

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    
    label = predict(image)
    st.write(f'Предсказанный класс: {label}')