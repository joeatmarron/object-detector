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
import random
import numpy as np
import platform
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

# Optional text-to-speech support with Eleven Labs
try:
    import requests
    import tempfile
    import subprocess
    import platform
    import threading
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        RETRY_AVAILABLE = True
    except ImportError:
        RETRY_AVAILABLE = False
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    RETRY_AVAILABLE = False
    print("Note: Text-to-speech dependencies not available. Install with: pip3 install requests")

# Load environment variables
load_dotenv()

# Constants
MODEL_ALTERNATIVES = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro-vision",
    "gemini-pro",
]

CATEGORY_EMOJI = {
    'Organic': '🍃',
    'Recyclables': '♻️',
    'Landfill': '🗑️',
    'Dangerous': '⚠️'
}

# Eleven Labs Voice IDs - Spanish-friendly voices for variety
# These voices work well with Spanish and support voice annotations
ELEVENLABS_VOICES = [
    "4enMglmEIYpK0bGn0QuZ",  # Rachel - English, works well with Spanish
    "g10k86KeEUyBqW9lcKYg",  # Adam - English
    "br0MPoLVxuslVxf61qHn",  # Bella - English
    "eRS3faIyd3KSRjzmhPxE",  # Antoni - English
    "iwNksRcTU0mglXb8PAk5",  # Elli - English
    "KuCuu213C5LmCbAvbEb8",  # Josh - English
    "wurf8bw1jlmweEM8XA4L",  # Arnold - English
    "tomkxGQGz4b1kE0EM722",  # Adam - English
    "Wl3O9lmFSMgGFTTwuS6f",  # Dorothy - English
    "lRf3yb6jZby4fn3q3Q7M",  # Charlotte - English
    "Ea3eYxXujQVVwq2KhqGC",  # Charlotte - English
    "eRALiEwGnmo3g1ze76Y2",  # Charlotte - English
    # Add more voices as needed - check Eleven Labs voice library for Spanish-specific voices
]

class TrashDetector:
    def __init__(self, api_key=None, model_name="gemini-2.0-flash-exp", enable_gpio=False, led_pin=18, buzzer_pin=None, enable_tts=False, elevenlabs_api_key=None, elevenlabs_voice_id=None, elevenlabs_model=None, language="es"):
        """
        Initialize the Trash Detector
        
        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
                       Common options: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-pro-vision
            enable_gpio: Enable GPIO hardware outputs (LED, buzzer, etc.)
            led_pin: GPIO pin number for LED (default: 18, BCM numbering)
            buzzer_pin: GPIO pin number for buzzer (optional, None to disable)
            enable_tts: Enable text-to-speech using Eleven Labs API
            elevenlabs_api_key: Eleven Labs API key (or set ELEVENLABS_API_KEY env var)
            elevenlabs_voice_id: Eleven Labs voice ID (default: "21m00Tcm4TlvDq8ikWAM" - Rachel)
            elevenlabs_model: Eleven Labs model ID (default: "eleven_turbo_v2_5" for low latency, or "eleven_v3" for better quality with annotations)
            language: Language code for prompts and responses (default: "es" for Spanish, "en" for English, etc.)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env file or pass as argument.")
        
        # Language support
        self.language = language or os.getenv("LANGUAGE", "es")
        
        # Category translations for different languages
        self.category_translations = {
            'es': {
                'Organic': 'ORGÁNICO',
                'Recyclables': 'INORGÁNICO',
                'Landfill': 'BASURA GENERAL',
                'Dangerous': 'PELIGROSO'
            },
            'en': {
                'Organic': 'ORGANIC',
                'Recyclables': 'RECYCLABLES',
                'Landfill': 'LANDFILL',
                'Dangerous': 'DANGEROUS'
            },
            # Add more languages as needed
            'fr': {
                'Organic': 'ORGANIQUE',
                'Recyclables': 'RECYCLABLES',
                'Landfill': 'DÉCHARGE',
                'Dangerous': 'DANGEREUX'
            },
            'de': {
                'Organic': 'ORGANISCH',
                'Recyclables': 'RECYCLING',
                'Landfill': 'DEPONIE',
                'Dangerous': 'GEFÄHRLICH'
            },
            'zh': {  # Chinese (Simplified)
                'Organic': '有机',
                'Recyclables': '可回收',
                'Landfill': '一般垃圾',
                'Dangerous': '危险'
            },
            'zh-tw': {  # Chinese (Traditional)
                'Organic': '有機',
                'Recyclables': '可回收',
                'Landfill': '一般垃圾',
                'Dangerous': '危險'
            }
        }
        
        # UI text translations
        self.ui_text = {
            'es': {
                'interactive_mode': 'MODO INTERACTIVO',
                'press_c_capture': "Presiona 'C' para capturar y analizar",
                'press_q_quit': "Presiona 'Q' o ESC para salir",
                'press_c_capture_q_quit': "Presiona 'C' para capturar, 'Q' para salir",
                'frame': 'Fotograma',
                'capturing': 'CAPTURANDO Y ANALIZANDO...',
                'analyzing': 'Analizando imagen con Gemini...',
                'trash_detected': '¡BASURA DETECTADA!',
                'no_trash': 'No hay basura'
            },
            'en': {
                'interactive_mode': 'INTERACTIVE MODE',
                'press_c_capture': "Press 'C' to capture and analyze",
                'press_q_quit': "Press 'Q' or ESC to quit",
                'press_c_capture_q_quit': "Press 'C' to capture, 'Q' to quit",
                'frame': 'Frame',
                'capturing': 'CAPTURING AND ANALYZING...',
                'analyzing': 'Analyzing image with Gemini...',
                'trash_detected': 'TRASH DETECTED!',
                'no_trash': 'No trash'
            },
            'fr': {
                'interactive_mode': 'MODE INTERACTIF',
                'press_c_capture': "Appuyez sur 'C' pour capturer et analyser",
                'press_q_quit': "Appuyez sur 'Q' ou ESC pour quitter",
                'press_c_capture_q_quit': "Appuyez sur 'C' pour capturer, 'Q' pour quitter",
                'frame': 'Image',
                'capturing': 'CAPTURE ET ANALYSE...',
                'analyzing': 'Analyse de l\'image avec Gemini...',
                'trash_detected': 'DÉCHET DÉTECTÉ!',
                'no_trash': 'Pas de déchet'
            },
            'de': {
                'interactive_mode': 'INTERAKTIVER MODUS',
                'press_c_capture': "Drücken Sie 'C' zum Erfassen und Analysieren",
                'press_q_quit': "Drücken Sie 'Q' oder ESC zum Beenden",
                'press_c_capture_q_quit': "Drücken Sie 'C' zum Erfassen, 'Q' zum Beenden",
                'frame': 'Bild',
                'capturing': 'ERFASSUNG UND ANALYSE...',
                'analyzing': 'Bildanalyse mit Gemini...',
                'trash_detected': 'MÜLL ERKANNT!',
                'no_trash': 'Kein Müll'
            },
            'zh': {  # Chinese (Simplified)
                'interactive_mode': '交互模式',
                'press_c_capture': "按 'C' 键捕获并分析",
                'press_q_quit': "按 'Q' 或 ESC 退出",
                'press_c_capture_q_quit': "按 'C' 捕获，'Q' 退出",
                'frame': '帧',
                'capturing': '正在捕获和分析...',
                'analyzing': '正在使用 Gemini 分析图像...',
                'trash_detected': '检测到垃圾！',
                'no_trash': '无垃圾'
            },
            'zh-tw': {  # Chinese (Traditional)
                'interactive_mode': '互動模式',
                'press_c_capture': "按 'C' 鍵捕獲並分析",
                'press_q_quit': "按 'Q' 或 ESC 退出",
                'press_c_capture_q_quit': "按 'C' 捕獲，'Q' 退出",
                'frame': '幀',
                'capturing': '正在捕獲和分析...',
                'analyzing': '正在使用 Gemini 分析圖像...',
                'trash_detected': '檢測到垃圾！',
                'no_trash': '無垃圾'
            }
        }
        
        # Get UI text for current language (fallback to English if not available)
        self.ui = self.ui_text.get(self.language, self.ui_text['en'])
        
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
        
        # Text-to-speech setup
        self.tts_enabled = enable_tts and TTS_AVAILABLE
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        
        # Voice selection: use provided voice ID, or rotating selection from available voices
        if elevenlabs_voice_id:
            self.elevenlabs_voice_id = elevenlabs_voice_id
            self.use_random_voice = False
            self.last_voice_id = None  # Not needed when using fixed voice
        else:
            env_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
            if env_voice_id:
                self.elevenlabs_voice_id = env_voice_id
                self.use_random_voice = False
                self.last_voice_id = None  # Not needed when using fixed voice
            else:
                # Use rotating voice selection from the voices array (always different)
                self.use_random_voice = True
                self.elevenlabs_voice_id = None  # Will be selected with rotation each time
                self.last_voice_id = None  # Track last used voice to ensure rotation
        
        # Default to faster model for lower latency (eleven_turbo_v2_5 is ~75ms vs ~200ms for eleven_v3)
        # Use eleven_v3 if you need full annotation support ([whispers], [giggles], etc.)
        self.elevenlabs_model = elevenlabs_model or os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
        
        # TTS setup
        if self.tts_enabled:
            if not self.elevenlabs_api_key:
                print("Warning: TTS enabled but ELEVENLABS_API_KEY not found. TTS disabled.")
                self.tts_enabled = False
            else:
                # Create HTTP session with connection pooling for better performance
                try:
                    self.tts_session = requests.Session()
                    if RETRY_AVAILABLE:
                        retry_strategy = Retry(
                            total=2,
                            backoff_factor=0.1,
                            status_forcelist=[429, 500, 502, 503, 504]
                        )
                        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
                        self.tts_session.mount("https://", adapter)
                except Exception:
                    # Fallback if urllib3 not available
                    self.tts_session = requests
                
                # Enable streaming for lower latency
                self.use_streaming = os.getenv("ELEVENLABS_STREAMING", "true").lower() == "true"
                
                if self.use_random_voice:
                    print(f"Text-to-speech enabled with rotating voice selection from {len(ELEVENLABS_VOICES)} voices (always different), model: {self.elevenlabs_model}, streaming: {self.use_streaming}")
                else:
                    print(f"Text-to-speech enabled with voice ID: {self.elevenlabs_voice_id}, model: {self.elevenlabs_model}, streaming: {self.use_streaming}")
        elif enable_tts and not TTS_AVAILABLE:
            print("Warning: TTS requested but dependencies not available. Install with: pip3 install requests")
        else:
            self.tts_session = None
    
    def _list_available_models(self):
        """List available models from the API"""
        try:
            models = self.client.models.list()
            model_list = list(models) if hasattr(models, '__iter__') else []
            return [getattr(m, 'name', str(m)) for m in model_list]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def _try_alternative_models(self, prompt, image, original_error):
        """Try alternative models with PIL image"""
        for alt_model in MODEL_ALTERNATIVES:
            if alt_model == self.model_name:
                continue
            try:
                print(f"Trying model: {alt_model}")
                response = self.client.models.generate_content(
                    model=alt_model,
                    contents=[prompt, image]
                )
                self.model_name = alt_model
                print(f"Successfully using model: {alt_model}")
                return response.text if hasattr(response, 'text') else str(response)
            except Exception:
                continue
        return None
    
    def _try_alternative_models_base64(self, prompt, image_bytes, original_error):
        """Try alternative models with base64 image"""
        for alt_model in MODEL_ALTERNATIVES:
            if alt_model == self.model_name:
                continue
            try:
                print(f"Trying model: {alt_model}")
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
                    response = self.client.models.generate_content(
                        model=alt_model,
                        contents=[prompt, {
                            "mime_type": "image/jpeg",
                            "data": image_bytes
                        }]
                    )
                self.model_name = alt_model
                print(f"Successfully using model: {alt_model}")
                return response.text if hasattr(response, 'text') else str(response)
            except Exception:
                continue
        return None
    
    def _display_results(self, results, results_path, width=50):
        """Display detection results in a formatted way"""
        print(f"\n{'='*width}")
        print("DETECTION RESULTS")
        print(f"{'='*width}")
        print(f"Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Trash Detected: {'YES' if results.get('detected') else 'NO'}")
        
        if results.get('trash_type'):
            trash_type = results.get('trash_type')
            if isinstance(trash_type, list):
                print(f"Trash Type: {', '.join(trash_type)}")
            else:
                print(f"Trash Type: {trash_type}")
        
        if results.get('category'):
            category = results.get('category')
            emoji = CATEGORY_EMOJI.get(category, '📦')
            print(f"Category: {emoji} {category}")
        
        if results.get('confidence'):
            print(f"Confidence: {results.get('confidence')}")
        
        if results.get('location') or results.get('location_description'):
            location = results.get('location') or results.get('location_description')
            print(f"Location: {location}")
        
        if results.get('error'):
            print(f"Error: {results.get('error')}")
        
        print(f"\nFull response saved to: {results_path}")
        print(f"{'='*width}\n")
        
    def _wrap_text(self, text, max_width=60):
        """Wrap text to fit within max_width characters per line"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            # If adding this word would exceed max_width, start a new line
            if current_length + word_length + len(current_line) > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _draw_utf8_text(self, img, text, position, font_scale, color, thickness, outline_color=(0, 0, 0), outline_thickness=2):
        """
        Draw UTF-8 text on OpenCV image using PIL for proper character encoding
        
        Args:
            img: OpenCV image (numpy array)
            text: Text to draw (can contain UTF-8 characters like á, é, í, ó, ú, ñ)
            position: (x, y) tuple for text position
            font_scale: Font size scale
            color: Text color (BGR tuple)
            thickness: Text thickness
            outline_color: Outline color (BGR tuple, default black)
            outline_thickness: Outline thickness
            
        Returns:
            Modified image
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            
            # Convert OpenCV image to PIL Image
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # Try to load a font that supports UTF-8 (including Chinese characters)
            try:
                system = platform.system()
                font_paths = []
                
                if system == "Darwin":  # macOS
                    # macOS fonts that support Chinese
                    font_paths = [
                        '/System/Library/Fonts/PingFang.ttc',  # PingFang (Chinese font on macOS)
                        '/System/Library/Fonts/STHeiti Light.ttc',  # STHeiti (Chinese font)
                        '/System/Library/Fonts/STSong.ttc',  # STSong (Chinese font)
                        '/System/Library/Fonts/Supplemental/Songti.ttc',  # Songti (Chinese font)
                        '/System/Library/Fonts/Supplemental/Kaiti.ttc',  # Kaiti (Chinese font)
                        '/Library/Fonts/Microsoft/SimHei.ttf',  # SimHei (if installed)
                        '/System/Library/Fonts/Helvetica.ttc',  # Fallback
                        '/System/Library/Fonts/Arial.ttf',  # Fallback
                    ]
                elif system == "Linux":
                    # Linux fonts that support Chinese
                    font_paths = [
                        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # WenQuanYi Micro Hei
                        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # WenQuanYi Zen Hei
                        '/usr/share/fonts/truetype/arphic/ukai.ttc',  # AR PL UKai
                        '/usr/share/fonts/truetype/arphic/uming.ttc',  # AR PL UMing
                        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK
                        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf',  # Noto Sans CJK (OpenType)
                        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Fallback
                        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',  # Fallback
                    ]
                elif system == "Windows":
                    # Windows fonts that support Chinese
                    font_paths = [
                        'C:/Windows/Fonts/simsun.ttc',  # SimSun (Chinese font)
                        'C:/Windows/Fonts/simhei.ttf',  # SimHei (Chinese font)
                        'C:/Windows/Fonts/msyh.ttc',  # Microsoft YaHei (Chinese font)
                        'C:/Windows/Fonts/simkai.ttf',  # SimKai (Chinese font)
                        'C:/Windows/Fonts/arial.ttf',  # Fallback
                    ]
                else:
                    # Generic fallback paths
                    font_paths = [
                        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                        '/System/Library/Fonts/Helvetica.ttc',
                        '/Windows/Fonts/arial.ttf',
                    ]
                
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        try:
                            # Calculate font size based on scale (approximate)
                            font_size = int(font_scale * 30)  # Adjust multiplier as needed
                            # For .ttc files, we may need to specify font index (0 is usually fine)
                            if font_path.endswith('.ttc'):
                                font = ImageFont.truetype(font_path, font_size, index=0)
                            else:
                                font = ImageFont.truetype(font_path, font_size)
                            print(f"Debug: Using font: {font_path}")
                            break
                        except Exception as e:
                            print(f"Debug: Failed to load font {font_path}: {e}")
                            continue
                
                if font is None:
                    # Fallback to default font
                    print("Debug: Using default font (may not support Chinese)")
                    font = ImageFont.load_default()
            except Exception as e:
                print(f"Debug: Font loading error: {e}")
                font = ImageFont.load_default()
            
            x, y = position
            
            # Draw outline first (draw multiple times for thicker outline)
            if outline_thickness > 0:
                for dx in range(-outline_thickness, outline_thickness + 1):
                    for dy in range(-outline_thickness, outline_thickness + 1):
                        if dx != 0 or dy != 0:
                            # Convert BGR to RGB for PIL
                            outline_rgb = (outline_color[2], outline_color[1], outline_color[0])
                            draw.text((x + dx, y + dy), text, font=font, fill=outline_rgb)
            
            # Draw main text (convert BGR to RGB for PIL)
            color_rgb = (color[2], color[1], color[0])
            draw.text((x, y), text, font=font, fill=color_rgb)
            
            # Convert back to OpenCV format
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img
            
        except ImportError:
            # Fallback to OpenCV putText if PIL not available (will show ? for special chars)
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + outline_thickness)
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            return img
        except Exception as e:
            # Fallback to OpenCV putText on any error
            print(f"Warning: Could not render UTF-8 text with PIL: {e}")
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, thickness + outline_thickness)
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            return img
    
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
    
    def _load_prompt(self, prompt_file=None):
        """
        Load prompt from file (language-specific if available)
        
        Args:
            prompt_file: Path to prompt file (default: prompt_{language}.txt or prompt.txt)
            
        Returns:
            str: Prompt text
        """
        if prompt_file is None:
            # Try language-specific prompt first, fallback to default
            lang_prompt_file = f"prompt_{self.language}.txt"
            default_prompt_file = "prompt.txt"
            
            # Check in the same directory as this script
            script_dir = Path(__file__).parent
            lang_path = script_dir / lang_prompt_file
            default_path = script_dir / default_prompt_file
            
            if lang_path.exists():
                prompt_file = lang_path
            elif default_path.exists():
                prompt_file = default_path
            else:
                raise FileNotFoundError(
                    f"Prompt file not found: {lang_prompt_file} or {default_prompt_file}\n"
                    f"Please create prompt_{self.language}.txt or prompt.txt in the project root directory."
                )
        else:
            prompt_file = Path(prompt_file)
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                if not prompt:
                    raise ValueError(f"Prompt file {prompt_file} is empty")
                print(f"Loaded prompt from: {prompt_file.name}")
                return prompt
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt from {prompt_file}: {e}")
    
    def detect_trash(self, image, prompt=None):
        """
        Use Gemini to detect trash in the image
        
        Args:
            image: numpy array image from OpenCV
            prompt: Custom prompt for detection (optional, if None loads from prompt.txt)
            
        Returns:
            dict: Detection results with confidence and details
        """
        if prompt is None:
            prompt = self._load_prompt()
        
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
                    response_text = self._try_alternative_models(prompt, pil_image, error_str)
                    if not response_text:
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
                    response_text = self._try_alternative_models_base64(prompt, image_bytes, error_str)
                    if not response_text:
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
                "category": None,
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
                
                # Check for trash detection in multiple possible formats
                detected = False
                # Try different possible keys from the API response (in order of likelihood)
                if "trash_detected" in parsed:
                    value = str(parsed["trash_detected"]).lower().strip()
                    detected = value.startswith("yes") or value == "true" or value == "1"
                elif "Is trash/litter detected?" in parsed:
                    value = str(parsed["Is trash/litter detected?"]).lower().strip()
                    detected = value.startswith("yes") or value == "true" or value == "1"
                elif "detected" in parsed:
                    value = str(parsed["detected"]).lower().strip()
                    detected = value in ["true", "yes", "1"]
                elif isinstance(parsed.get("detected"), bool):
                    detected = parsed["detected"]
                
                # If still not detected, check if trash_type exists (indicates trash was found)
                if not detected and parsed.get("trash_type"):
                    # If trash_type has content, trash was detected
                    trash_type = parsed.get("trash_type")
                    if isinstance(trash_type, list) and len(trash_type) > 0:
                        detected = True
                    elif isinstance(trash_type, str) and trash_type.strip():
                        detected = True
                
                result["detected"] = detected
                    
            except (json.JSONDecodeError, KeyError):
                # If JSON parsing fails, extract info from text
                text_lower = response_text.lower()
                # Look for positive indicators
                has_yes = "yes" in text_lower
                has_trash = "trash" in text_lower or "litter" in text_lower
                has_no = "no" in text_lower and ("trash" not in text_lower and "litter" not in text_lower)
                
                # If we see "yes" with trash/litter, it's detected
                # If we see "no trash" or "no litter", it's not detected
                if has_yes and has_trash:
                    result["detected"] = True
                elif has_no:
                    result["detected"] = False
                else:
                    # Default: if trash/litter mentioned, assume detected
                    result["detected"] = has_trash
            
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
        
        self._display_results(results, results_path, width=50)
        
        # Trigger hardware outputs if enabled
        if self.gpio_enabled:
            self.trigger_hardware_output(results.get('detected', False))
        
        # Play text-to-speech for what_to_do if available (async to not block)
        if results.get('what_to_do'):
            self._text_to_speech(results.get('what_to_do'), async_mode=True)
        
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
        print(self.ui['interactive_mode'])
        print("="*60)
        print(self.ui['press_c_capture'])
        print(self.ui['press_q_quit'])
        print("="*60 + "\n")
        
        # Create output directories
        Path(output_dir).mkdir(exist_ok=True)
        captures_dir = os.path.join(output_dir, "captures")
        Path(captures_dir).mkdir(exist_ok=True)
        results_dir = os.path.join(output_dir, "results")
        Path(results_dir).mkdir(exist_ok=True)
        
        frame_count = 0
        
        # Track flash state for non-blocking colored screens
        flash_state = {
            'active': False,
            'start_time': None,
            'duration': 0,
            'color': None,
            'type': None  # 'green' for capture, 'category' for category flash
        }
        
        # Track results display state for non-blocking overlay
        results_overlay = {
            'active': False,
            'start_time': None,
            'duration': 20.0,  # Show results for 20 seconds (longer for accessibility)
            'data': None  # Will store results dict
        }
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Display the frame
                display_frame = frame.copy()
                
                # Check if we're in a flash period and overlay color if needed
                current_time = datetime.now().timestamp()
                if flash_state['active'] and flash_state['start_time']:
                    elapsed = current_time - flash_state['start_time']
                    if elapsed < flash_state['duration']:
                        # Still in flash period - overlay the color
                        overlay = display_frame.copy()
                        overlay[:] = flash_state['color']
                        # Blend overlay with original (optional: you can make it fully opaque)
                        display_frame = overlay
                    else:
                        # Flash period ended
                        flash_state['active'] = False
                        flash_state['start_time'] = None
                
                # Check if we should show results overlay
                if results_overlay['active'] and results_overlay['start_time']:
                    elapsed = current_time - results_overlay['start_time']
                    if elapsed < results_overlay['duration'] and results_overlay['data']:
                        # Draw results overlay on frame - CENTERED for accessibility
                        results = results_overlay['data']
                        height, width = display_frame.shape[:2]
                        
                        if results.get('detected'):
                            # Get category for display
                            category = results.get('category', '')
                            
                            # Translate category based on language
                            lang_translations = self.category_translations.get(self.language, self.category_translations['es'])
                            category_display = lang_translations.get(category, category.upper() if category else '')
                            
                            # Calculate text positions for centering
                            # Main category text (LARGE for accessibility)
                            text_height = 0  # Initialize for use in what_to_do positioning
                            if category and category_display:
                                category_text = category_display  # Spanish category name, already uppercase
                                # Use different colors for different categories (high contrast)
                                category_colors = {
                                    'Organic': (0, 255, 0),      # Green (BGR)
                                    'Recyclables': (255, 255, 255), # White (BGR) - high contrast
                                    'Landfill': (255, 255, 255),    # White (BGR) - high contrast
                                    'Dangerous': (0, 0, 255)      # Red (BGR)
                                }
                                category_color = category_colors.get(category, (255, 255, 255))
                                
                                # Calculate text size and position for centering
                                font_scale = 3.0  # Much larger for accessibility
                                thickness = 5
                                (text_width, text_height), baseline = cv2.getTextSize(
                                    category_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                                )
                                
                                # Center horizontally and vertically
                                x_pos = (width - text_width) // 2
                                y_pos = (height + text_height) // 2  # Center vertically
                                
                                # Draw text with outline for better visibility (using UTF-8 compatible method)
                                display_frame = self._draw_utf8_text(
                                    display_frame, category_text, (x_pos, y_pos), 
                                    font_scale, category_color, thickness,
                                    outline_color=(0, 0, 0), outline_thickness=2
                                )
                            
                            # Add what_to_do_display if available (centered, smaller text below category)
                            # Use what_to_do_display (without annotations) for display, fallback to what_to_do
                            what_to_do_display = results.get('what_to_do_display') or results.get('what_to_do', '')
                            if what_to_do_display:
                                # Wrap text to fit on screen (approximately 60 characters per line for centered text)
                                wrapped_lines = self._wrap_text(what_to_do_display, max_width=60)
                                
                                # Calculate starting y position (below category text)
                                font_scale_instruction = 1.2  # Larger than before for accessibility
                                thickness_instruction = 3
                                line_height = 50  # More spacing between lines
                                
                                # Start below the category text (or center if no category)
                                if category:
                                    start_y = (height + text_height) // 2 + 100
                                else:
                                    start_y = height // 2 + 50
                                
                                for i, line in enumerate(wrapped_lines):
                                    (line_width, line_height_text), _ = cv2.getTextSize(
                                        line, cv2.FONT_HERSHEY_SIMPLEX, font_scale_instruction, thickness_instruction
                                    )
                                    line_x = (width - line_width) // 2  # Center each line
                                    line_y = start_y + (i * line_height)
                                    
                                    # Draw text with UTF-8 support
                                    display_frame = self._draw_utf8_text(
                                        display_frame, line, (line_x, line_y),
                                        font_scale_instruction, (255, 255, 255), thickness_instruction,
                                        outline_color=(0, 0, 0), outline_thickness=2
                                    )
                        else:
                            result_text = self.ui['no_trash'].upper()
                            color = (0, 255, 0)  # Green
                            
                            # Center the "No trash" text
                            font_scale = 2.0
                            thickness = 4
                            (text_width, text_height), baseline = cv2.getTextSize(
                                result_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                            )
                            x_pos = (width - text_width) // 2
                            y_pos = (height + text_height) // 2
                            
                            # Draw with UTF-8 support
                            display_frame = self._draw_utf8_text(
                                display_frame, result_text, (x_pos, y_pos),
                                font_scale, color, thickness,
                                outline_color=(0, 0, 0), outline_thickness=2
                            )
                            
                            # Show what_to_do_display even when no trash is detected
                            # Use what_to_do_display (without annotations) for display, fallback to what_to_do
                            what_to_do_display = results.get('what_to_do_display') or results.get('what_to_do', '')
                            if what_to_do_display:
                                wrapped_lines = self._wrap_text(what_to_do_display, max_width=60)
                                
                                font_scale_instruction = 1.2
                                thickness_instruction = 3
                                line_height = 50
                                start_y = y_pos + 100
                                
                                for i, line in enumerate(wrapped_lines):
                                    (line_width, line_height_text), _ = cv2.getTextSize(
                                        line, cv2.FONT_HERSHEY_SIMPLEX, font_scale_instruction, thickness_instruction
                                    )
                                    line_x = (width - line_width) // 2
                                    line_y = start_y + (i * line_height)
                                    
                                    # Draw text with UTF-8 support
                                    display_frame = self._draw_utf8_text(
                                        display_frame, line, (line_x, line_y),
                                        font_scale_instruction, (255, 255, 255), thickness_instruction,
                                        outline_color=(0, 0, 0), outline_thickness=2
                                    )
                    else:
                        # Results overlay period ended
                        results_overlay['active'] = False
                        results_overlay['start_time'] = None
                        results_overlay['data'] = None
                
                cv2.putText(display_frame, self.ui['press_c_capture_q_quit'], 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"{self.ui['frame']}: {frame_count}", 
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
                    # Flash green screen for visual feedback (non-blocking)
                    flash_state['active'] = True
                    flash_state['start_time'] = datetime.now().timestamp()
                    flash_state['duration'] = 0.1  # 100ms
                    flash_state['color'] = (0, 255, 0)  # Green (BGR)
                    flash_state['type'] = 'green'
                    
                    print("\n" + "-"*60)
                    print(self.ui['capturing'])
                    print("-"*60)
                    
                    # Capture current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(captures_dir, f"capture_{timestamp}.jpg") if save_image else None
                    
                    if save_image:
                        cv2.imwrite(image_path, frame)
                        print(f"Image saved to: {image_path}")
                    
                    # Analyze the frame
                    print(self.ui['analyzing'])
                    results = self.detect_trash(frame)
                    
                    # Save results to results subdirectory
                    results_path = os.path.join(results_dir, f"results_{timestamp}.json")
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Display results
                    self._display_results(results, results_path, width=60)
                    
                    # Trigger hardware outputs if enabled
                    if self.gpio_enabled:
                        self.trigger_hardware_output(results.get('detected', False))
                    
                    # Show category-based color flash for visual feedback (non-blocking)
                    category = results.get('category')
                    if category:
                        # Define colors in BGR format
                        category_colors = {
                            'Organic': (0, 165, 255),      # Orange (BGR)
                            'Recyclables': (255, 0, 0),    # Blue (BGR)
                            'Landfill': (128, 128, 128),  # Gray (BGR)
                            'Dangerous': (0, 0, 255)       # Red (BGR) - for dangerous items
                        }
                        flash_color = category_colors.get(category, (128, 128, 128))
                        # Start non-blocking flash
                        flash_state['active'] = True
                        flash_state['start_time'] = datetime.now().timestamp()
                        flash_state['duration'] = 20.0  # Match results overlay duration (20 seconds) so color stays while text is visible
                        flash_state['color'] = flash_color
                        flash_state['type'] = 'category'
                    
                    # Start non-blocking results overlay
                    results_overlay['active'] = True
                    results_overlay['start_time'] = datetime.now().timestamp()
                    results_overlay['data'] = results
                    
                    # Play text-to-speech if available (non-blocking)
                    if results.get('what_to_do'):
                        if self.tts_enabled:
                            print(f"Debug: Playing TTS for: {results.get('what_to_do')[:50]}...")
                            self._text_to_speech(results.get('what_to_do'), async_mode=True)
                        else:
                            print("Debug: TTS is disabled. Use --tts flag to enable.")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyAllWindows()
    
    def _text_to_speech(self, text, async_mode=False):
        """
        Convert text to speech using Eleven Labs API with latency optimizations
        
        Args:
            text: Text to convert to speech (should be in Spanish for what_to_do)
            async_mode: If True, generate audio in background thread (non-blocking)
        """
        if not self.tts_enabled:
            print("Debug: TTS is not enabled. Use --tts flag and set ELEVENLABS_API_KEY.")
            return
        if not text:
            print("Debug: No text provided for TTS.")
            return
        
        if async_mode:
            # Generate audio in background thread to avoid blocking
            thread = threading.Thread(target=self._text_to_speech_sync, args=(text,), daemon=True)
            thread.start()
        else:
            self._text_to_speech_sync(text)
    
    def _text_to_speech_sync(self, text):
        """Synchronous TTS generation with optimizations"""
        try:
            # Select voice: use configured voice or rotating selection (always different)
            if self.use_random_voice:
                # Get available voices excluding the last one used
                available_voices = [v for v in ELEVENLABS_VOICES if v != self.last_voice_id]
                
                # If all voices were used or only one voice available, use all voices
                if not available_voices:
                    available_voices = ELEVENLABS_VOICES
                
                # Select a different voice from available ones
                voice_id = random.choice(available_voices)
                self.last_voice_id = voice_id  # Remember this voice for next time
            else:
                voice_id = self.elevenlabs_voice_id
            
            # Use streaming endpoint for lower latency (starts playing while generating)
            if self.use_streaming:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
            else:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            # Optimized settings for lower latency:
            # - Lower stability = faster generation (but less consistent)
            # - Lower similarity_boost = faster (but less voice accuracy)
            # - Use speaker_boost for better quality at lower settings
            # Note: Stability must be one of [0.0, 0.5, 1.0] (0.0=Creative/fast, 0.5=Natural, 1.0=Robust/slow)
            data = {
                "text": text,
                "model_id": self.elevenlabs_model,  # Use turbo model for speed
                "voice_settings": {
                    "stability": 0.0,  # 0.0 = Creative (fastest), 0.5 = Natural (balanced), 1.0 = Robust (slowest)
                    "similarity_boost": 0.5,  # Keep at 0.5 for good voice accuracy
                    "style": 0.0,
                    "use_speaker_boost": True
                },
                "output_format": "mp3_44100_128"  # Optimized format: 44.1kHz, 128kbps (good quality, reasonable size)
            }
            
            # Use session for connection pooling (faster subsequent requests)
            session = self.tts_session if self.tts_session else requests
            response = session.post(url, json=data, headers=headers, timeout=15, stream=self.use_streaming)
            
            if response.status_code == 200:
                if self.use_streaming:
                    # Streaming mode: play audio as it arrives (lower latency)
                    self._play_streaming_audio(response)
                else:
                    # Non-streaming: save to file then play
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name
                    
                    self._play_audio_file(tmp_path)
                    
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            else:
                error_msg = f"Warning: Eleven Labs API error: {response.status_code} - {response.text}"
                print(error_msg)
                print(f"Debug: TTS request failed. Check your ELEVENLABS_API_KEY and voice ID.")
        except Exception as e:
            error_msg = f"Warning: Text-to-speech failed: {e}"
            print(error_msg)
            import traceback
            print(f"Debug: TTS error details: {traceback.format_exc()}")
    
    def _play_streaming_audio(self, response):
        """Play streaming audio as it arrives (lower latency)"""
        try:
            system = platform.system()
            
            # For streaming, we need to pipe directly to audio player
            if system == "Linux":
                # Use mpg123 with stdin for streaming (lowest latency)
                for player in ["mpg123", "mpg321"]:
                    try:
                        process = subprocess.Popen(
                            [player, "-"],  # "-" means read from stdin
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        # Stream audio chunks to player
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                process.stdin.write(chunk)
                        process.stdin.close()
                        process.wait()
                        break
                    except FileNotFoundError:
                        continue
                    except Exception:
                        # Fallback to file-based if streaming fails
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    tmp_file.write(chunk)
                            tmp_path = tmp_file.name
                        self._play_audio_file(tmp_path)
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                        break
            else:
                # For macOS/Windows, fallback to file-based (streaming not easily supported)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                    tmp_path = tmp_file.name
                
                self._play_audio_file(tmp_path)
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        except Exception as e:
            print(f"Warning: Could not play streaming audio: {e}")
    
    def _play_audio_file(self, file_path):
        """Play audio file using system audio player"""
        try:
            system = platform.system()
            if system == "Linux":
                # Try multiple audio players common on Linux/Raspberry Pi
                audio_played = False
                for player in ["mpg123", "mpg321", "ffplay", "aplay"]:
                    try:
                        result = subprocess.run([player, file_path], 
                                              stdout=subprocess.DEVNULL, 
                                              stderr=subprocess.DEVNULL,
                                              check=False,
                                              timeout=30)
                        if result.returncode == 0:
                            audio_played = True
                            print(f"Debug: Audio played successfully using {player}")
                            break
                    except FileNotFoundError:
                        continue
                    except subprocess.TimeoutExpired:
                        print(f"Warning: Audio player {player} timed out")
                        continue
                
                if not audio_played:
                    print("Warning: No audio player found. Install one of: mpg123, mpg321, ffplay, or aplay")
                    print("  On Ubuntu/Debian: sudo apt-get install mpg123")
                    print("  On Raspberry Pi: sudo apt-get install mpg123")
            elif system == "Darwin":  # macOS
                result = subprocess.run(["afplay", file_path], check=False, timeout=30)
                if result.returncode == 0:
                    print("Debug: Audio played successfully using afplay")
                else:
                    print(f"Warning: afplay returned error code: {result.returncode}")
            elif system == "Windows":
                result = subprocess.run(["start", file_path], shell=True, check=False, timeout=30)
                if result.returncode == 0:
                    print("Debug: Audio played successfully")
                else:
                    print(f"Warning: Audio playback returned error code: {result.returncode}")
        except subprocess.TimeoutExpired:
            print("Warning: Audio playback timed out")
        except Exception as e:
            print(f"Warning: Could not play audio: {e}")
            import traceback
            print(f"Debug: Audio playback error details: {traceback.format_exc()}")
    
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
        
        # Close TTS session
        if hasattr(self, 'tts_session') and self.tts_session and hasattr(self.tts_session, 'close'):
            self.tts_session.close()


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
    parser.add_argument("--tts", action="store_true", help="Enable text-to-speech using Eleven Labs API")
    parser.add_argument("--elevenlabs-api-key", type=str, help="Eleven Labs API key (or set ELEVENLABS_API_KEY env var)")
    parser.add_argument("--elevenlabs-voice-id", type=str, help="Eleven Labs voice ID (default: Rachel)")
    parser.add_argument("--elevenlabs-model", type=str, help="Eleven Labs model ID (default: eleven_turbo_v2_5 for low latency, or eleven_v3 for better quality)")
    parser.add_argument("--language", "-l", type=str, default="es", 
                       help="Language code for prompts and responses (default: es for Spanish, en for English, fr for French, de for German, zh for Chinese Simplified, zh-tw for Chinese Traditional)")
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
            buzzer_pin=args.buzzer_pin,
            enable_tts=args.tts,
            elevenlabs_api_key=args.elevenlabs_api_key,
            elevenlabs_voice_id=args.elevenlabs_voice_id,
            elevenlabs_model=args.elevenlabs_model,
            language=args.language
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

