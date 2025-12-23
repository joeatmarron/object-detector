#!/usr/bin/env python3
"""
Simple script to test if the camera is working
"""

import cv2
import sys

def test_camera(camera_index=0):
    """Test camera at given index"""
    print(f"Testing camera at index {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera at index {camera_index}")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"❌ Failed to capture frame from camera {camera_index}")
        cap.release()
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Camera {camera_index} is working!")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frame captured successfully")
    
    # Save a test image
    cv2.imwrite("test_camera.jpg", frame)
    print(f"   Test image saved to: test_camera.jpg")
    
    cap.release()
    return True

if __name__ == "__main__":
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    if not test_camera(camera_index):
        print("\n💡 Tips:")
        print("   - Check camera connections")
        print("   - Try different camera indices: python3 test_camera.py 0, 1, 2, etc.")
        print("   - For Raspberry Pi Camera Module, ensure it's enabled in raspi-config")
        sys.exit(1)
    else:
        sys.exit(0)

