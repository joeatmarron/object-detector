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
from dotenv import load_dotenv

# Use google.genai package (required)
try:
    import google.genai as genai
except ImportError:
    raise ImportError("google.genai package is required. Install with: pip3 install google-genai")

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
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
                       Common options: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-pro-vision
            enable_gpio: Enable GPIO hardware outputs (LED, buzzer, etc.)
            led_pin: GPIO pin number for LED (default: 18, BCM numbering)
            buzzer_pin: GPIO pin number for buzzer (optional, None to disable)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env file or pass as argument.")
        
        # Initialize google.genai client
        self.model_name = model_name
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            print(f"Using google.genai package with model: {model_name}")
            # Model validation will happen when we try to use it
        except Exception as e:
            raise RuntimeError(f"Failed to initialize google.genai client: {e}")
        
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
    
    def _list_available_models(self):
        """List available models from the API"""
        try:
            models = self.client.models.list()
            model_list = list(models) if hasattr(models, '__iter__') else []
            return [getattr(m, 'name', str(m)) for m in model_list]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
        
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
        
        # Convert image to format compatible with Gemini API
        # Try PIL Image first (preferred)
        try:
            from PIL import Image
            # Convert OpenCV image (BGR) to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Call google.genai API - use proper Content structure
            try:
                # The API expects Content objects with Parts
                # Try importing types first
                try:
                    from google.genai.types import Content, Part
                    # Create Content with Parts
                    content = Content(
                        parts=[
                            Part(text=prompt),
                            Part(inline_data=pil_image)
                        ]
                    )
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[content]
                    )
                except (ImportError, AttributeError, TypeError) as type_error:
                    # If types don't work, try simple format
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[prompt, pil_image]
                    )
                
                # Extract text from response
                response_text = response.text if hasattr(response, 'text') else str(response)
            except Exception as api_error:
                error_str = str(api_error)
                # If model not found, try alternatives
                if "404" in error_str or "not found" in error_str.lower():
                    print(f"Model '{self.model_name}' not found. Trying alternatives...")
                    alternatives = [
                        "gemini-2.0-flash-exp",
                        "gemini-1.5-flash",
                        "gemini-1.5-pro",
                        "gemini-pro-vision",
                        "gemini-pro",
                    ]
                    
                    for alt_model in alternatives:
                        if alt_model == self.model_name:
                            continue
                        try:
                            print(f"Trying model: {alt_model}")
                            response = self.client.models.generate_content(
                                model=alt_model,
                                contents=[prompt, pil_image]
                            )
                            self.model_name = alt_model
                            print(f"Successfully using model: {alt_model}")
                            response_text = response.text if hasattr(response, 'text') else str(response)
                            break
                        except Exception:
                            continue
                    else:
                        # All alternatives failed
                        raise RuntimeError(
                            f"Failed to find a working model. "
                            f"Original error: {error_str[:200]}. "
                            f"Use --list-models to see available models."
                        )
                else:
                    # Other error, re-raise
                    raise api_error
                
        except (ImportError, Exception) as e:
            # Fallback to base64 format
            image_base64 = self.image_to_base64(image)
            image_bytes = base64.b64decode(image_base64)
            
            # Try different base64 formats for google.genai API
            try:
                # Try with Content and Part objects
                try:
                    from google.genai.types import Content, Part, Blob
                    blob = Blob(data=image_bytes, mime_type="image/jpeg")
                    content = Content(
                        parts=[
                            Part(text=prompt),
                            Part(inline_data=blob)
                        ]
                    )
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[content]
                    )
                except (ImportError, AttributeError, TypeError):
                    # Try creating a File from bytes
                    try:
                        import io
                        from google.genai import File
                        # Create a file-like object from bytes
                        image_file = io.BytesIO(image_bytes)
                        uploaded_file = self.client.files.upload(path=None, file=image_file)
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=[prompt, uploaded_file]
                        )
                    except Exception:
                        # Last resort: try dict format
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=[{
                                "parts": [
                                    {"text": prompt},
                                    {
                                        "inline_data": {
                                            "mime_type": "image/jpeg",
                                            "data": image_bytes
                                        }
                                    }
                                ]
                            }]
                        )
                
                response_text = response.text if hasattr(response, 'text') else str(response)
            except Exception as api_error:
                error_str = str(api_error)
                # If model not found, try alternatives
                if "404" in error_str or "not found" in error_str.lower():
                    print(f"Model '{self.model_name}' not found. Trying alternatives...")
                    alternatives = [
                        "gemini-2.0-flash-exp",
                        "gemini-1.5-flash",
                        "gemini-1.5-pro",
                        "gemini-pro-vision",
                        "gemini-pro",
                    ]
                    
                    for alt_model in alternatives:
                        if alt_model == self.model_name:
                            continue
                        try:
                            print(f"Trying model: {alt_model}")
                            # Try with Content/Part first
                            try:
                                from google.genai.types import Content, Part, Blob
                                blob = Blob(data=image_bytes, mime_type="image/jpeg")
                                content = Content(
                                    parts=[
                                        Part(text=prompt),
                                        Part(inline_data=blob)
                                    ]
                                )
                                response = self.client.models.generate_content(
                                    model=alt_model,
                                    contents=[content]
                                )
                            except (ImportError, AttributeError, TypeError):
                                # Fallback to simple format
                                response = self.client.models.generate_content(
                                    model=alt_model,
                                    contents=[prompt, {
                                        "mime_type": "image/jpeg",
                                        "data": image_bytes
                                    }]
                                )
                            
                            self.model_name = alt_model
                            print(f"Successfully using model: {alt_model}")
                            response_text = response.text if hasattr(response, 'text') else str(response)
                            break
                        except Exception:
                            continue
                    else:
                        # All alternatives failed
                        return {
                            "timestamp": datetime.now().isoformat(),
                            "error": f"Failed to find a working model. Original error: {error_str[:200]}",
                            "detected": False
                        }
                else:
                    # Other error
                    return {
                        "timestamp": datetime.now().isoformat(),
                        "error": f"Failed to call google.genai API: {error_str[:200]}",
                        "detected": False
                    }
        
        try:
            result = {
                "timestamp": datetime.now().isoformat(),
                "raw_response": response_text,
                "detected": False,
                "trash_type": None,
                "confidence": None,
                "location": None,
                "recommendations": None
            }
            
            # Try to parse JSON from response
            try:
                # Extract JSON from response if it's wrapped in markdown
                text = response_text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(text)
                result.update(parsed)
                result["detected"] = parsed.get("Is trash/litter detected?", "").lower().startswith("yes")
            except (json.JSONDecodeError, KeyError):
                # If JSON parsing fails, extract info from text
                text_lower = response_text.lower()
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
        # Create output directories
        Path(output_dir).mkdir(exist_ok=True)
        captures_dir = os.path.join(output_dir, "captures")
        Path(captures_dir).mkdir(exist_ok=True)
        results_dir = os.path.join(output_dir, "results")
        Path(results_dir).mkdir(exist_ok=True)
        
        # Capture image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(captures_dir, f"capture_{timestamp}.jpg") if save_image else None
        
        print("Capturing image...")
        image = self.capture_image(image_path)
        
        # Detect trash
        print("Analyzing image with Gemini...")
        results = self.detect_trash(image)
        
        # Save results to results subdirectory
        results_path = os.path.join(results_dir, f"results_{timestamp}.json")
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
    
    def run_interactive_mode(self, output_dir="output", save_image=True):
        """
        Run interactive mode - show camera feed and capture on 'C' keypress
        
        Args:
            output_dir: Directory to save images and results
            save_image: Whether to save captured images
            
        Controls:
            'C' or 'c': Capture and analyze current frame
            'Q' or 'q' or ESC: Quit
        """
        if not self.camera:
            raise RuntimeError("Camera not initialized. Call initialize_camera() first.")
        
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("Press 'C' to capture and analyze")
        print("Press 'Q' or ESC to quit")
        print("="*60 + "\n")
        
        # Create output directories
        Path(output_dir).mkdir(exist_ok=True)
        captures_dir = os.path.join(output_dir, "captures")
        Path(captures_dir).mkdir(exist_ok=True)
        results_dir = os.path.join(output_dir, "results")
        Path(results_dir).mkdir(exist_ok=True)
        
        frame_count = 0
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Display the frame
                display_frame = frame.copy()
                cv2.putText(display_frame, "Press 'C' to capture, 'Q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Frame: {frame_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Trash Detector - Interactive Mode", display_frame)
                
                # Wait for keypress (1ms delay to allow window to update)
                key = cv2.waitKey(1) & 0xFF
                
                # Check for quit
                if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                    print("\nQuitting interactive mode...")
                    break
                
                # Check for capture
                if key == ord('c') or key == ord('C'):
                    print("\n" + "-"*60)
                    print("CAPTURING AND ANALYZING...")
                    print("-"*60)
                    
                    # Capture current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(captures_dir, f"capture_{timestamp}.jpg") if save_image else None
                    
                    if save_image:
                        cv2.imwrite(image_path, frame)
                        print(f"Image saved to: {image_path}")
                    
                    # Analyze the frame
                    print("Analyzing image with Gemini...")
                    results = self.detect_trash(frame)
                    
                    # Save results to results subdirectory
                    results_path = os.path.join(results_dir, f"results_{timestamp}.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Display results
                    print(f"\n{'='*60}")
                    print("DETECTION RESULTS")
                    print(f"{'='*60}")
                    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
                    print(f"Trash Detected: {'YES' if results.get('detected') else 'NO'}")
                    if results.get('trash_type'):
                        print(f"Trash Type: {results.get('trash_type')}")
                    if results.get('confidence'):
                        print(f"Confidence: {results.get('confidence')}")
                    if results.get('location'):
                        print(f"Location: {results.get('location')}")
                    if results.get('error'):
                        print(f"Error: {results.get('error')}")
                    print(f"\nFull response saved to: {results_path}")
                    print(f"{'='*60}\n")
                    
                    # Trigger hardware outputs if enabled
                    if self.gpio_enabled:
                        self.trigger_hardware_output(results.get('detected', False))
                    
                    # Show result on frame briefly
                    result_text = "TRASH DETECTED!" if results.get('detected') else "No trash"
                    color = (0, 0, 255) if results.get('detected') else (0, 255, 0)
                    cv2.putText(display_frame, result_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.imshow("Trash Detector - Interactive Mode", display_frame)
                    cv2.waitKey(2000)  # Show result for 2 seconds
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyAllWindows()
    
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
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-exp", 
                       help="Gemini model name (default: gemini-2.0-flash-exp). Use --list-models to see available models.")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for images and results")
    parser.add_argument("--no-save", action="store_true", help="Don't save captured images")
    parser.add_argument("--gpio", action="store_true", help="Enable GPIO hardware outputs (LED, buzzer)")
    parser.add_argument("--led-pin", type=int, default=18, help="GPIO pin for LED (BCM numbering, default: 18)")
    parser.add_argument("--buzzer-pin", type=int, default=None, help="GPIO pin for buzzer (optional)")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive mode: show camera feed, press 'C' to capture, 'Q' to quit")
    
    args = parser.parse_args()
    
    try:
        # If --list-models is specified, list available models and exit
        if args.list_models:
            print("Fetching available models...")
            # Create a minimal detector just to access the API
            detector = TrashDetector(api_key=args.api_key, model_name="gemini-1.5-flash")
            models = detector._list_available_models()
            if models:
                print("\nAvailable models:")
                for model in models:
                    print(f"  - {model}")
            else:
                print("Could not retrieve model list. Check your API key.")
            sys.exit(0)
        
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
        
        # Run in interactive mode or single capture mode
        if args.interactive:
            detector.run_interactive_mode(
                output_dir=args.output_dir,
                save_image=not args.no_save
            )
            detector.cleanup()
            sys.exit(0)
        else:
            # Single capture mode
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

