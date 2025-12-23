#!/bin/bash
# Setup script for Trash Detector on Raspberry Pi

echo "Setting up Trash Detector..."

# Check Python version
python3 --version || { echo "Python 3 is required but not installed. Aborting."; exit 1; }

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "# Google Gemini API Key" > .env
    echo "# Get your API key from: https://makersuite.google.com/app/apikey" >> .env
    echo "GEMINI_API_KEY=your_api_key_here" >> .env
    echo "Please edit .env and add your actual API key!"
else
    echo ".env file already exists"
fi

# Create output directory
mkdir -p output

echo "Setup complete!"
echo "Next steps:"
echo "1. Edit .env and add your Gemini API key"
echo "2. Test the camera: python3 -c 'import cv2; cap = cv2.VideoCapture(0); print(\"Camera works!\" if cap.isOpened() else \"Camera failed\"); cap.release()'"
echo "3. Run the detector: python3 trash_detector.py"

