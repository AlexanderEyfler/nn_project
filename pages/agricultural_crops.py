import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

# Загрузите сохранённое состояние модели
model_path = 'models/danila/model_50.pth'
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 30)  # 30 - количество классов в задаче
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Функция для предсказания
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    classes = ['tomato', 'pineapple', 'Tobacco-plant', 'jowar', 'Olive-tree', 'Fox_nut(Makhana)', 
             'banana', 'sunflower', 'Cherry', 'Lemon', 'Cucumber', 'Coffee-plant', 'vigna-radiati(Mung)', 
           'cardamom', 'cotton', 'maize', 'sugarcane', 'papaya', 'gram', 'mustard-oil', 'Pearl_millet(bajra)', 
           'chilli', 'tea', 'jute', 'coconut', 'clove', 'wheat', 'soyabean', 'almond', 'rice']
    return classes[predicted.item()]

# Интерфейс Streamlit
st.title('Классификация изображений с помощью ResNet50')

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    
    label = predict(image)
    st.write(f'Предсказанный класс: {label}')