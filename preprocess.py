from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_image(file):
    image = Image.open(file).convert("RGB")
    return transform(image).unsqueeze(0)