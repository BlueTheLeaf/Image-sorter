# Image Sorter

Type in the image prompt, and the program will try to find the best matches from your image collection!

## Features
- Search for images based on text prompts.
- View the top 5 most relevant images with thumbnails.
- Directly access the image files' directories.

## Installation

1. **Clone or Download the Repository:**
   Ensure you have the project folder on your local machine.

2. **Install Dependencies:**
   Make sure you have Python installed. Install the required libraries by running:
   {YOU KNOW WHAT}
   pip install torch torchvision transformers pillow scikit-learn
   {YOU KNOW WHAT}

## Setup

1. **Prepare the Images Folder:**
   - Create a folder named `images` in the same directory as `main.py`.
   - Place all the images you want to search through in this folder.

2. **Run the Program:**
   - Execute the `main.py` file:
```
     python main.py
```
   - Enter a descriptive text prompt when prompted.

## Usage

1. **Launch the Program:**
   - Run `main.py` to start the program.

2. **Enter Your Search Prompt:**
   - Type in the description of the image you're looking for (e.g., "sunset over mountains").

3. **Wait for Results:**
   - The program will process the images and display the top 5 matches based on your prompt. Note that processing time may vary depending on the number of images.

4. **View Results:**
   - The top 5 matching images will be displayed with thumbnails, similarity scores, and direct links to their directories.

## Example

1. After running `main.py`, you might enter a prompt like `"waffle picture"`.
2. The program will show the top 5 images that best match your description, including a preview and directory links.

## Troubleshooting

- **If the Program Is Slow:**
  - Processing time increases with the number of images. Consider using a smaller set for faster results.

- **Image Size Warning:**
  - If you encounter warnings about large images, resize them before adding them to the folder.

## Contributions

Feel free to contribute by submitting issues or pull requests!
