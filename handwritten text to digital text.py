import cv2
import numpy as np
import pytesseract
from PIL import Image
import argparse
import os

def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy for handwritten text.
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Noise removal using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Dilate to connect components
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    return dilated

def recognize_text(preprocessed_img):
    """
    Perform OCR on the preprocessed image.
    """
    # Configure pytesseract for handwritten text
    custom_config = r'--oem 3 --psm 6 -l eng'
    
    # Convert OpenCV image to PIL format
    pil_img = Image.fromarray(preprocessed_img)
    
    # Perform OCR
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    
    return text

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Convert handwritten text in images to digital text")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("-o", "--output", help="Path to output text file (optional)")
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return
    
    print(f"Processing image: {args.image_path}")
    
    try:
        # Preprocess the image
        preprocessed = preprocess_image(args.image_path)
        
        # Recognize text
        text = recognize_text(preprocessed)
        
        # Print extracted text
        print("\nExtracted Text:")
        print("-" * 50)
        print(text)
        print("-" * 50)
        
        # Save to file if output path is provided
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as file:
                file.write(text)
            print(f"Text saved to: {args.output}")
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()