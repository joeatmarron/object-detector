# Hardware Setup Guide

This guide explains how to connect hardware outputs to your Raspberry Pi for the trash detector.

## Hardware Outputs Supported

The trash detector can control:
- **LED** - Visual indicator when trash is detected
- **Buzzer/Piezo** - Audio alert when trash is detected
- **Other GPIO devices** - Can be extended for servos, displays, etc.

## GPIO Pin Layout

The code uses **BCM (Broadcom) pin numbering** (not physical pin numbers).

Common GPIO pins used:
- **GPIO 18 (Pin 12)** - Default LED pin (PWM capable)
- **GPIO 23 (Pin 16)** - Good for buzzer
- **GPIO 24 (Pin 18)** - Alternative LED pin
- **GPIO 25 (Pin 22)** - Alternative buzzer pin

### Raspberry Pi GPIO Pinout Reference

```
    3.3V  [1]  [2]  5V
  GPIO2  [3]  [4]  5V
  GPIO3  [5]  [6]  GND
  GPIO4  [7]  [8]  GPIO14
    GND  [9]  [10] GPIO15
 GPIO17 [11] [12] GPIO18  ← Default LED pin
 GPIO27 [13] [14] GND
 GPIO22 [15] [16] GPIO23  ← Good for buzzer
    3.3V [17] [18] GPIO24
 GPIO10 [19] [20] GND
  GPIO9  [21] [22] GPIO25
 GPIO11 [23] [24] GPIO8
    GND  [25] [26] GPIO7
```

## Wiring Instructions

### LED Setup

**Components needed:**
- 1x LED (any color, 3mm or 5mm)
- 1x 220Ω resistor (or 330Ω for brighter LEDs)
- Jumper wires

**Wiring:**
1. Connect the **anode (long leg)** of LED to GPIO 18 (Pin 12) via the resistor
2. Connect the **cathode (short leg)** of LED to GND (any ground pin)
3. The resistor should be between GPIO pin and LED anode

```
GPIO 18 (Pin 12) → [220Ω Resistor] → LED Anode (+)
LED Cathode (-) → GND
```

**Alternative pin:** Use `--led-pin 24` to use GPIO 24 instead

### Buzzer Setup

**Components needed:**
- 1x Active buzzer (or passive buzzer with transistor)
- Jumper wires
- Optional: 1kΩ resistor (for active buzzer)

**Wiring (Active Buzzer):**
1. Connect **positive leg** to GPIO 23 (Pin 16) via resistor
2. Connect **negative leg** to GND

```
GPIO 23 (Pin 16) → [1kΩ Resistor] → Buzzer (+)
Buzzer (-) → GND
```

**Note:** Active buzzers have polarity (+/-), passive buzzers may need a transistor circuit.

## Usage

### Enable Hardware Outputs

Run the detector with GPIO enabled:

```bash
# LED only (default pin 18)
python3 trash_detector.py --gpio

# LED on custom pin
python3 trash_detector.py --gpio --led-pin 24

# LED + Buzzer
python3 trash_detector.py --gpio --led-pin 18 --buzzer-pin 23
```

### Behavior

When trash is detected:
- **LED**: Turns ON for 2 seconds
- **Buzzer**: Plays 3 short beeps (if enabled)

When no trash is detected:
- All outputs remain OFF

## Testing Hardware

Test your GPIO setup without running full detection:

```python
import RPi.GPIO as GPIO
import time

# Test LED on pin 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

print("LED ON")
GPIO.output(18, GPIO.HIGH)
time.sleep(2)

print("LED OFF")
GPIO.output(18, GPIO.LOW)
GPIO.cleanup()
```

## Safety Notes

⚠️ **Important:**
- Always use current-limiting resistors with LEDs (220Ω-330Ω recommended)
- Don't exceed 3.3V on GPIO pins (they're 3.3V logic, not 5V tolerant)
- Use appropriate resistors for buzzers to limit current
- Double-check pin numbers before connecting (BCM vs physical numbering)
- Never short GPIO pins directly to power or ground

## Extending for More Hardware

You can easily extend the code to control:
- **Servo motors** - For automated trash collection mechanisms
- **Relays** - For controlling higher voltage devices
- **LCD displays** - For showing detection status
- **Stepper motors** - For camera positioning

Example extension in code:
```python
# In trigger_hardware_output method, add:
if detected:
    # Control servo, relay, etc.
    servo_pin.angle(90)  # Example
```

## Troubleshooting

**LED not lighting up:**
- Check resistor value (too high = dim, too low = may damage LED)
- Verify LED polarity (anode to GPIO, cathode to GND)
- Test with multimeter to verify GPIO pin is outputting 3.3V
- Try different GPIO pin

**Buzzer not working:**
- Verify it's an active buzzer (or add transistor circuit for passive)
- Check polarity (+ to GPIO, - to GND)
- Test with multimeter
- Try different GPIO pin

**Permission errors:**
- Run with `sudo` (not recommended for production)
- Add user to `gpio` group: `sudo usermod -a -G gpio $USER`
- Log out and back in after adding to group

