import torch
from transformers import ResNetModel
from PIL import Image
from torchvision.transforms import transforms
from metrics import *

# Define the path to the saved model weights
saved_weights_path = "path/to/saved_weights.pth"

# Create an instance of the model
model = Model()

# Load the saved weights
model.load_state_dict(torch.load(saved_weights_path))
model.to(DEVICE)
model.eval()

# Define a function to generate a caption for a given image
def generate_caption(image_path, model):
    chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    NUM_CHAR = len(chars)
    idx2char = {k: v for k, v in enumerate(chars)}
    char2idx = {v: k for k, v in enumerate(chars)}

    img = Image.open(image_path).convert('RGB')

    # Apply the same image processing as in the training data
    img_proc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = img_proc(img).unsqueeze(0).to(DEVICE)

    # Generate caption
    with torch.no_grad():
        output = model(img)
        output = output.permute(0, 2, 1).contiguous().view(-1, NUM_CHAR)
        _, predicted_indices = torch.max(output, 1)

    # Convert indices to text
    predicted_caption = ''.join([idx2char[idx.item()] for idx in predicted_indices])

    return predicted_caption

# Example usage:
image_path = "path/to/your/image.jpg"  # Replace with the path to your test image
predicted_caption = generate_caption(image_path, model)
print("Predicted Caption:", predicted_caption)
