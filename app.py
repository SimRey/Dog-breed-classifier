import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import io


import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Load your model
class VGG19(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(VGG19, self).__init__()
        # Load pre-trained VGG19 model
        self.vgg19 = models.vgg19(pretrained=pretrained)
        # Replace the last fully connected layer
        if pretrained:
            # Freeze all layers except the final fully connected layer
            for param in self.vgg19.parameters():
                param.requires_grad = False
        self.vgg19.classifier[6] = nn.Linear(self.vgg19.classifier[6].in_features, num_classes)

    def forward(self, x):
        x = self.vgg19(x)
        return F.log_softmax(x, dim=1)

# Test the network
model = VGG19(num_classes=120, pretrained=True)
model.load_state_dict(torch.load("Dogsbreed.pt"))
model.eval()

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
