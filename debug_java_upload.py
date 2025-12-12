"""
Debug script to test the exact format Java is sending
This will help identify what's wrong with the file upload
"""
import requests
import os

def test_raw_bytes_upload():
    """Test sending raw bytes like Java does"""

    print("="*60)
    print("Testing Raw Bytes Upload (Simulating Java)")
    print("="*60)

    # Find a test image
    test_images = [
        'my_test_images/brain_scan.jpg',
        'data/binary/tumor/Not Cancer  (1).jpg',
        'data/binary/no_tumor/Not Cancer  (1).jpg',
    ]

    test_image = None
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break

    if not test_image:
        print("❌ No test image found!")
        return

    print(f"\n📁 Test image: {test_image}")

    # Read file bytes
    with open(test_image, 'rb') as f:
        file_bytes = f.read()

    print(f"📊 File size: {len(file_bytes)} bytes")
    print(f"🔍 First 16 bytes (hex): {file_bytes[:16].hex()}")

    # Check magic bytes
    if file_bytes.startswith(b'\xff\xd8\xff'):
        print("✅ Valid JPEG signature detected")
    elif file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        print("✅ Valid PNG signature detected")
    else:
        print("⚠️  Unknown image format")

    # Test 1: Send with proper content type
    print("\n" + "="*60)
    print("Test 1: With proper content-type (image/jpeg)")
    print("="*60)

    files = {
        'file': ('brain-scan.jpg', file_bytes, 'image/jpeg')
    }

    try:
        response = requests.post('http://localhost:5000/api/predict', files=files)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS!")
            result = response.json()
            print(f"Prediction: {result.get('prediction')}")
        else:
            print(f"❌ FAILED: {response.text}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    # Test 2: Send with octet-stream (like Java)
    print("\n" + "="*60)
    print("Test 2: With application/octet-stream (Java default)")
    print("="*60)

    files = {
        'file': ('brain-scan.jpg', file_bytes, 'application/octet-stream')
    }

    try:
        response = requests.post('http://localhost:5000/api/predict', files=files)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS!")
            result = response.json()
            print(f"Prediction: {result.get('prediction')}")
        else:
            print(f"❌ FAILED: {response.text}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    # Test 3: Send with no content type
    print("\n" + "="*60)
    print("Test 3: With no content-type specified")
    print("="*60)

    files = {
        'file': ('brain-scan.jpg', file_bytes)
    }

    try:
        response = requests.post('http://localhost:5000/api/predict', files=files)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS!")
            result = response.json()
            print(f"Prediction: {result.get('prediction')}")
        else:
            print(f"❌ FAILED: {response.text}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    # Test 4: Send raw bytes in different way
    print("\n" + "="*60)
    print("Test 4: Using BytesIO wrapper")
    print("="*60)

    from io import BytesIO

    files = {
        'file': ('brain-scan.jpg', BytesIO(file_bytes), 'application/octet-stream')
    }

    try:
        response = requests.post('http://localhost:5000/api/predict', files=files)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS!")
            result = response.json()
            print(f"Prediction: {result.get('prediction')}")
        else:
            print(f"❌ FAILED: {response.text}")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def inspect_debug_file():
    """Inspect the debug file saved by Flask"""
    debug_file = 'temp_debug_image.bin'

    if not os.path.exists(debug_file):
        print(f"\n⚠️  No debug file found at {debug_file}")
        return

    print("\n" + "="*60)
    print("Inspecting Debug File")
    print("="*60)

    with open(debug_file, 'rb') as f:
        data = f.read()

    print(f"File size: {len(data)} bytes")
    print(f"First 32 bytes (hex): {data[:32].hex()}")
    print(f"First 32 bytes (raw): {data[:32]}")

    # Check if it's a valid image
    if data.startswith(b'\xff\xd8\xff'):
        print("✅ Valid JPEG signature")
    elif data.startswith(b'\x89PNG\r\n\x1a\n'):
        print("✅ Valid PNG signature")
    elif data.startswith(b'BM'):
        print("✅ Valid BMP signature")
    else:
        print("❌ Invalid or unknown image format")

    # Try to open with PIL
    from PIL import Image
    from io import BytesIO

    try:
        img = Image.open(BytesIO(data))
        print(f"✅ PIL can open it: format={img.format}, size={img.size}, mode={img.mode}")
    except Exception as e:
        print(f"❌ PIL cannot open it: {e}")


if __name__ == '__main__':
    print("\n🔬 Java Backend Debug Test\n")

    # Check if server is running
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code == 200:
            print("✅ Server is running\n")
        else:
            print("⚠️  Server responded but not healthy\n")
    except:
        print("❌ Server is not running! Start it with: python api_server.py\n")
        exit(1)

    # Run tests
    test_raw_bytes_upload()

    # Inspect debug file if it exists
    inspect_debug_file()

    print("\n" + "="*60)
    print("Debug Complete")
    print("="*60)

