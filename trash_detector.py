#!/usr/bin/env python3
"""
Trash Detector using Google Gemini Vision API
Designed for Raspberry Pi with camera module
"""

import os
import sys
import cv2
import base64
import json
from datetime import datetime
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Optional GPIO support for Raspberry Pi hardware outputs
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Note: RPi.GPIO not available. Hardware outputs disabled. Install with: pip3 install RPi.GPIO")

# Load environment variables
load_dotenv()

class TrashDetector:
    def __init__(self, api_key=None, model_name="gemini-1.5-flash", enable_gpio=False, led_pin=18, buzzer_pin=None):
        """
        Initialize the Trash Detector
        
        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
            model_name: Gemini model to use (default: gemini-1.5-flash)
            enable_gpio: Enable GPIO hardware outputs (LED, buzzer, etc.)
            led_pin: GPIO pin number for LED (default: 18, BCM numbering)
            buzzer_pin: GPIO pin number for buzzer (optional, None to disable)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env file or pass as argument.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.camera = None
        
        # GPIO setup
        self.gpio_enabled = enable_gpio and GPIO_AVAILABLE
        self.led_pin = led_pin
        self.buzzer_pin = buzzer_pin
        
        if self.gpio_enabled:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.led_pin, GPIO.OUT)
            GPIO.output(self.led_pin, GPIO.LOW)
            
            if self.buzzer_pin:
                GPIO.setup(self.buzzer_pin, GPIO.OUT)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
            
            print(f"GPIO enabled - LED on pin {self.led_pin}" + 
                  (f", Buzzer on pin {self.buzzer_pin}" if self.buzzer_pin else ""))
        elif enable_gpio and not GPIO_AVAILABLE:
            print("Warning: GPIO requested but RPi.GPIO not available. Install with: pip3 install RPi.GPIO")
        
    def initialize_camera(self, camera_index=0):
        """
        Initialize the camera
        
        Args:
            camera_index: Camera index (0 for default, or specific USB camera)
        """
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera at index {camera_index}")
        
        # Set camera resolution (adjust based on your camera)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"Camera initialized successfully")
        
    def capture_image(self, save_path=None):
        """
        Capture an image from the camera
        
        Args:
            save_path: Optional path to save the captured image
            
        Returns:
            numpy array: Image as numpy array
        """
        if not self.camera:
            raise RuntimeError("Camera not initialized. Call initialize_camera() first.")
        
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera")
        
        if save_path:
            cv2.imwrite(save_path, frame)
            print(f"Image saved to {save_path}")
        
        return frame
    
    def image_to_base64(self, image):
        """
        Convert OpenCV image to base64 string
        
        Args:
            image: numpy array image from OpenCV
            
        Returns:
            str: Base64 encoded image
        """
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def detect_trash(self, image, prompt=None):
        """
        Use Gemini to detect trash in the image
        
        Args:
            image: numpy array image from OpenCV
            prompt: Custom prompt for detection (optional)
            
        Returns:
            dict: Detection results with confidence and details
        """
        if prompt is None:
            prompt = """Analyze this image and determine if there is trash or litter visible.
            
Please provide:
1. Is trash/litter detected? (Yes/No)
2. What type of trash is visible? (e.g., plastic bottle, wrapper, can, etc.)
3. Confidence level (High/Medium/Low)
4. Location description (e.g., center, left side, ground, etc.)
5. Any recommendations for cleanup?

Format your response as JSON with these fields."""
        
        # Convert image to base64
        image_base64 = self.image_to_base64(image)
        
        # Prepare the image for Gemini
        image_data = {
            "mime_type": "image/jpeg",
            "data": base64.b64decode(image_base64)
        }
        
        try:
            # Call Gemini API
            response = self.model.generate_content([prompt, image_data])
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "raw_response": response.text,
                "detected": False,
                "trash_type": None,
                "confidence": None,
                "location": None,
                "recommendations": None
            }
            
            # Try to parse JSON from response
            try:
                # Extract JSON from response if it's wrapped in markdown
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(text)
                result.update(parsed)
                result["detected"] = parsed.get("Is trash/litter detected?", "").lower().startswith("yes")
            except (json.JSONDecodeError, KeyError):
                # If JSON parsing fails, extract info from text
                text_lower = response.text.lower()
                result["detected"] = "yes" in text_lower or "trash" in text_lower or "litter" in text_lower
            
            return result
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "detected": False
            }
    
    def run_detection(self, save_image=True, output_dir="output"):
        """
        Run a complete detection cycle: capture image and analyze
        
        Args:
            save_image: Whether to save the captured image
            output_dir: Directory to save images and results
            
        Returns:
            dict: Detection results
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Capture image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(output_dir, f"capture_{timestamp}.jpg") if save_image else None
        
        print("Capturing image...")
        image = self.capture_image(image_path)
        
        # Detect trash
        print("Analyzing image with Gemini...")
        results = self.detect_trash(image)
        
        # Save results
        results_path = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*50}")
        print("DETECTION RESULTS")
        print(f"{'='*50}")
        print(f"Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Trash Detected: {'YES' if results.get('detected') else 'NO'}")
        if results.get('trash_type'):
            print(f"Trash Type: {results.get('trash_type')}")
        if results.get('confidence'):
            print(f"Confidence: {results.get('confidence')}")
        if results.get('location'):
            print(f"Location: {results.get('location')}")
        print(f"\nFull response saved to: {results_path}")
        print(f"{'='*50}\n")
        
        # Trigger hardware outputs if enabled
        if self.gpio_enabled:
            self.trigger_hardware_output(results.get('detected', False))
        
        return results
    
    def trigger_hardware_output(self, detected, duration=2.0):
        """
        Trigger hardware outputs based on detection result
        
        Args:
            detected: Boolean indicating if trash was detected
            duration: How long to keep outputs active (seconds)
        """
        if not self.gpio_enabled:
            return
        
        import time
        
        if detected:
            # Trash detected - turn on LED and buzzer
            print("🔴 Activating hardware outputs: LED ON, Buzzer ON")
            GPIO.output(self.led_pin, GPIO.HIGH)
            
            if self.buzzer_pin:
                # Beep pattern: 3 short beeps
                for _ in range(3):
                    GPIO.output(self.buzzer_pin, GPIO.HIGH)
                    time.sleep(0.2)
                    GPIO.output(self.buzzer_pin, GPIO.LOW)
                    time.sleep(0.1)
            
            time.sleep(duration)
            GPIO.output(self.led_pin, GPIO.LOW)
        else:
            # No trash - quick green flash (if you have a green LED on another pin)
            # Or just keep LED off
            print("🟢 No trash detected - hardware outputs off")
    
    def cleanup(self):
        """Release camera and GPIO resources"""
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()
            print("Camera released")
        
        if self.gpio_enabled:
            GPIO.output(self.led_pin, GPIO.LOW)
            if self.buzzer_pin:
                GPIO.output(self.buzzer_pin, GPIO.LOW)
            GPIO.cleanup()
            print("GPIO cleaned up")


def main():
    """Main function to run the trash detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trash Detector using Gemini Vision API")
    parser.add_argument("--api-key", type=str, help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for images and results")
    parser.add_argument("--no-save", action="store_true", help="Don't save captured images")
    parser.add_argument("--gpio", action="store_true", help="Enable GPIO hardware outputs (LED, buzzer)")
    parser.add_argument("--led-pin", type=int, default=18, help="GPIO pin for LED (BCM numbering, default: 18)")
    parser.add_argument("--buzzer-pin", type=int, default=None, help="GPIO pin for buzzer (optional)")
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = TrashDetector(
            api_key=args.api_key, 
            model_name=args.model,
            enable_gpio=args.gpio,
            led_pin=args.led_pin,
            buzzer_pin=args.buzzer_pin
        )
        
        # Initialize camera
        print("Initializing camera...")
        detector.initialize_camera(camera_index=args.camera)
        
        # Run detection
        results = detector.run_detection(
            save_image=not args.no_save,
            output_dir=args.output_dir
        )
        
        # Cleanup
        detector.cleanup()
        
        # Exit with appropriate code
        sys.exit(0 if not results.get('detected') else 1)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

