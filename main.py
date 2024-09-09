import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_features(prompt):
    """Convert text prompt to feature vector."""
    inputs = processor(text=prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.numpy()

def get_image_features(image_path):
    """Convert image to feature vector."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.numpy()

def get_image_files_from_main_folder(main_folder):
    """Get all image files from the main folder and its subdirectories."""
    image_files = []
    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def find_top_matches(prompt, main_folder, top_n=5):
    """Find the top N images in the folder and its subdirectories that best match the text prompt."""
    text_features = get_text_features(prompt)
    matches = []

    image_files = get_image_files_from_main_folder(main_folder)

    for image_path in image_files:
        try:
            image_features = get_image_features(image_path)
            # Calculate similarity
            score = cosine_similarity(text_features, image_features)[0][0]
            matches.append((image_path, score))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # Sort matches by score in descending order
    matches.sort(key=lambda x: x[1], reverse=True)

    # Get the top N matches
    top_matches = matches[:top_n]

    return top_matches

# Example usage
print("----------------------------")
prompt = input("What are you searching for? ")
main_folder = './images'  # Main folder containing images and subdirectories

top_matches = find_top_matches(prompt, main_folder, top_n=5)

if top_matches:
    for i, (image_path, score) in enumerate(top_matches):
        print(f"Rank {i+1}: {image_path} ({score * 100:.2f}%)")
else:
    print("No matching images found.")
