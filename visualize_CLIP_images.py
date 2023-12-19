import torch
import clip
from PIL import Image


def generate_caption(image_path, model_path):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_path, device=device)

    # Process the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Generate a caption
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(clip.tokenize(["a photo of a", "a drawing of a"]))

        # Calculate similarities and choose the best caption
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarities[0].topk(1)

        return "a photo of a" if indices[0] == 0 else "a drawing of a"

# Example usage
caption = generate_caption("path_to_your_image.jpg", "path_to_your_model.pth")
print(caption)