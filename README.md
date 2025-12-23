# Trash Detector for Raspberry Pi

A simple trash detection system using Google Gemini Vision API and a camera module. This project is designed to run on a Raspberry Pi and can detect trash/litter in captured images.

## Features

- 📸 Camera capture using OpenCV (supports Raspberry Pi Camera Module and USB cameras)
- 🤖 AI-powered trash detection using Google Gemini Vision API
- 💾 Automatic saving of captured images and detection results
- 📊 JSON output with detailed detection information
- 🔧 Easy configuration via environment variables
- 🔌 **GPIO hardware outputs** - Control LEDs, buzzers, and other devices when trash is detected

## Prerequisites

- Raspberry Pi (any model with camera support)
- Raspberry Pi Camera Module or USB webcam
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- **Optional:** LED and/or buzzer for hardware outputs (see [HARDWARE_SETUP.md](HARDWARE_SETUP.md))

## Installation

### 1. Clone or download this repository

```bash
cd /path/to/detector
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

For Raspberry Pi, you might need to install system dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-pip libopencv-dev
```

### 3. Set up environment variables

Copy the example environment file and add your API key:

```bash
cp .env.example .env
nano .env
```

Add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

## Usage

### Basic Usage

Simply run the detector script:

```bash
python3 trash_detector.py
```

This will:
1. Initialize the camera
2. Capture an image
3. Analyze it with Gemini
4. Save the image and results to the `output/` directory
5. Display results in the terminal

### Command Line Options

```bash
python3 trash_detector.py --help
```

Available options:
- `--api-key`: Override API key from environment variable
- `--camera`: Camera index (default: 0, use 1, 2, etc. for USB cameras)
- `--model`: Gemini model name (default: gemini-1.5-flash)
- `--output-dir`: Output directory (default: output)
- `--no-save`: Don't save captured images

### Examples

**Use a specific camera:**
```bash
python3 trash_detector.py --camera 1
```

**Use a different Gemini model:**
```bash
python3 trash_detector.py --model gemini-1.5-pro
```

**Don't save images:**
```bash
python3 trash_detector.py --no-save
```

**Enable hardware outputs (LED/buzzer):**
```bash
# LED only (default pin 18)
python3 trash_detector.py --gpio

# LED + Buzzer
python3 trash_detector.py --gpio --led-pin 18 --buzzer-pin 23
```

See [HARDWARE_SETUP.md](HARDWARE_SETUP.md) for detailed wiring instructions.

## Raspberry Pi Setup

### Enable Camera Module

If using the official Raspberry Pi Camera Module:

```bash
sudo raspi-config
```

Navigate to: `Interface Options` → `Camera` → `Enable`

### Test Camera

Test if your camera works:

```bash
# For Raspberry Pi Camera Module
libcamera-hello -t 0

# Or test with Python
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works!' if cap.isOpened() else 'Camera failed'); cap.release()"
```

### USB Camera

If using a USB webcam, it should work out of the box. You may need to specify the camera index:

```bash
python3 trash_detector.py --camera 0  # Try 0, 1, 2, etc.
```

## Output

The detector creates two files for each detection:

1. **Image file**: `output/capture_YYYYMMDD_HHMMSS.jpg` - The captured image
2. **Results file**: `output/results_YYYYMMDD_HHMMSS.json` - Detection results in JSON format

Example JSON output:
```json
{
  "timestamp": "2024-01-15T10:30:45",
  "detected": true,
  "trash_type": "plastic bottle",
  "confidence": "High",
  "location": "center of frame",
  "raw_response": "..."
}
```

## Integration

### Scheduled Detection

You can set up a cron job to run detection periodically:

```bash
crontab -e
```

Add a line to run every hour:
```
0 * * * * cd /path/to/detector && /usr/bin/python3 trash_detector.py
```

### Web Interface (Future Enhancement)

The code can be easily extended to create a web interface using Flask or FastAPI.

### Automation

The script exits with code 0 if no trash is detected, and code 1 if trash is detected. This allows for automation:

```bash
python3 trash_detector.py && echo "No trash found" || echo "Trash detected!"
```

## Troubleshooting

### Camera not found
- Check camera connections
- Try different camera indices: `--camera 0`, `--camera 1`, etc.
- For Raspberry Pi Camera Module, ensure it's enabled in `raspi-config`

### API Key errors
- Verify your API key in the `.env` file
- Check that the Gemini API is enabled in your Google Cloud Console
- Ensure you have internet connectivity

### Import errors
- Make sure all dependencies are installed: `pip3 install -r requirements.txt`
- For OpenCV on Raspberry Pi, you might need: `sudo apt-get install python3-opencv`

## License

This project is open source and available for modification and distribution.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

