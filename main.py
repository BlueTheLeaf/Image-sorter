import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

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

def show_results(results):
    """Display the results in a GUI."""
    root = tk.Tk()
    root.title("Top Matches")
    root.geometry("800x600")  # Set a default size for the window

    # Create a canvas for scrolling
    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    
    # Create a frame inside the canvas
    frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    # Update the scrollregion after adding widgets
    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    for i, (image_path, score) in enumerate(results):
        item_frame = ttk.Frame(frame, padding=5, relief="solid", borderwidth=1)
        item_frame.grid(row=i, column=0, sticky="ew", padx=10, pady=10)

        # Display the thumbnail
        img = Image.open(image_path)
        img.thumbnail((150, 150))  # Resize the thumbnail
        img_tk = ImageTk.PhotoImage(img)
        label_img = ttk.Label(item_frame, image=img_tk)
        label_img.image = img_tk  # Keep a reference to avoid garbage collection
        label_img.grid(row=0, column=0)

        # Display the file path
        label_text = ttk.Label(item_frame, text=f"{os.path.basename(image_path)}\nScore: {score * 100:.2f}%", anchor="w")
        label_text.grid(row=1, column=0, sticky="w")

        # Display the directory link
        button_link = ttk.Button(item_frame, text="Open Directory", command=lambda p=image_path: open_directory(p))
        button_link.grid(row=2, column=0, pady=(5, 0))

    # Update the scrollregion after adding widgets
    frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    root.mainloop()

def open_directory(image_path):
    """Open the directory containing the image."""
    dir_path = os.path.dirname(image_path)
    abs_dir_path = os.path.abspath(dir_path)  # Convert to absolute path
    if os.path.exists(abs_dir_path):
        try:
            os.startfile(abs_dir_path)
        except Exception as e:
            print(f"Failed to open directory: {e}")
    else:
        print(f"Directory not found: {abs_dir_path}")

# Example usage
prompt = input("What are you searching for? ")
main_folder = './images'  # Main folder containing images and subdirectories

top_matches = find_top_matches(prompt, main_folder, top_n=5)

if top_matches:
    show_results(top_matches)
else:
    print("No matching images found.")