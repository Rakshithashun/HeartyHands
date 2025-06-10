import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import google.generativeai as genai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class_names = [
    'FreshApple', 'FreshBanana', 'FreshBellpepper', 'FreshCarrot', 'FreshCucumber',
    'FreshMango', 'FreshOrange', 'FreshPotato', 'FreshStrawberry', 'FreshTomato',
    'RottenApple', 'RottenBanana', 'RottenBellpepper', 'RottenCarrot', 'RottenCucumber',
    'RottenMango', 'RottenOrange', 'RottenPotato', 'RottenStrawberry', 'RottenTomato'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
num_classes = len(class_names)
model = FoodClassifier(num_classes).to(device)
model.load_state_dict(torch.load("food_classifier.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Gemini API
GENAI_API_KEY = "AIzaSyCWo3UtC0gyJ5gsFw7mfRHTFiVRUKIEGh8"
genai.configure(api_key=GENAI_API_KEY)

def get_gemini_suggestions(prediction):
    prompt = f"Provide only ways to use all parts of {prediction} effectively to minimize food waste, without any additional information."
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text if response else "No suggestions available."

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]
    except Exception as e:
        return f"Error processing image: {str(e)}"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Combined index route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('zero-waste.html', prediction="No file uploaded", suggestions="", image=None)

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = predict_image(filepath)
        suggestions = get_gemini_suggestions(prediction)
        image_url = f"/uploads/{filename}"

        return render_template('zero-waste.html', prediction=f"Predicted: {prediction}", suggestions=suggestions, image=image_url)

    return render_template('zero-waste.html', prediction=None, suggestions="", image=None)

if __name__ == '__main__':
    app.run(debug=True)
