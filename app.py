import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
from torchvision import transforms
import io

# Load your model
model = torch.load("C:/Users/Lenovo/Desktop/Python/Deep Learning/PyTorch/3. Convolutional Neural Networks/Exercises/Dog breed classifier/Dogsbreed.pt", map_location=torch.device('cpu'))

transform = transforms.Compose([
        transforms.Resize(224),             
        transforms.CenterCrop(224),         
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

def predict(image):
    image = transform(image)
    image = image.view(1,3,224,224)
    with torch.no_grad():
        predicted_label = model(image).argmax()
    return predicted_label.item()

# Define a function to get breed name (modify this based on your labels)
def get_breed_name(label):
    
    df = pd.read_csv("labels.csv")
    le = LabelEncoder()
    df["breed"] = le.fit_transform(df["breed"])

    dog_breeds = le.classes_
    return dog_breeds[label]

# Streamlit interface
st.title("Dog Breed Classifier")
st.write("Upload an image of a dog to classify its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(io.BytesIO(uploaded_file.read()))

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the breed
    label = predict(image)
    breed_name = get_breed_name(label)
    st.write(f'The predicted breed is: {breed_name}')
