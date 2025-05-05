from fastapi import FastAPI, File, UploadFile
import numpy as np
import torch
from fastapi.middleware.cors import CORSMiddleware
from segmentation_models_pytorch import Unet
import io
from fastapi.responses import JSONResponse
import cv2
from PIL import Image
from torch import nn
import torchvision.models as models
from torchvision import datasets, transforms
import torch.nn.functional as F

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the final layer for your number of classes
num_classes = 2  # This must match the number of classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('F:\\final_project\skin\skin_cancer_model'))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*'])

# Prediction endpoint
@app.post('/predictions')
async def pred(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Load image and preprocess
    image = Image.open(io.BytesIO(contents)).convert("RGB")  # Convert to RGB if grayscale
    image_tensor = transform(image)
    image_tensor.unsqueeze_(0)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        label = None
        if predicted_class.item() == 0:
            label = 'Benign'
            
        else:
            label = 'Malignant'
                
    
    # Return the prediction result
    return JSONResponse(content={"prediction": label})

# To run the FastAPI app, use the command: uvicorn main:app --reload