"""
Test script to verify Java backend compatibility with Flask API
This simulates how a Java backend would send files to the Flask server
"""
import requests
import os
import sys

def test_file_upload(image_path, api_url='http://localhost:5000/api/predict'):
    """
    Test file upload to the API (simulating Java backend multipart upload)

    Args:
        image_path: Path to the image file
        api_url: API endpoint URL
    """
    print(f"\n{'='*60}")
    print(f"Testing File Upload: {image_path}")
    print(f"API URL: {api_url}")
    print(f"{'='*60}")

    if not os.path.exists(image_path):
        print(f"❌ ERROR: File not found: {image_path}")
        return False

    try:
        # Open file and send as multipart/form-data (like Java would do)
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}

            print(f"📤 Sending file: {os.path.basename(image_path)}")
            print(f"   File size: {os.path.getsize(image_path)} bytes")

            response = requests.post(api_url, files=files)

            print(f"\n📥 Response Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ SUCCESS!")
                print(f"\n📊 Results:")
                print(f"   Prediction: {result.get('prediction', 'N/A')}")
                print(f"   Has Tumor: {result.get('has_tumor', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 0)*100:.2f}%")

                if result.get('has_tumor'):
                    print(f"\n🔬 Tumor Type Analysis:")
                    print(f"   Type: {result.get('tumor_type', 'N/A')}")
                    print(f"   Confidence: {result.get('tumor_type_confidence', 0)*100:.2f}%")
                    print(f"\n   Probabilities:")
                    for tumor_type, prob in result.get('tumor_type_probabilities', {}).items():
                        print(f"      {tumor_type}: {prob*100:.2f}%")

                return True
            else:
                print(f"❌ ERROR: Request failed")
                print(f"   Response: {response.text}")
                return False

    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Cannot connect to server at {api_url}")
        print(f"   Make sure the Flask server is running: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base64_upload(image_path, api_url='http://localhost:5000/api/predict'):
    """
    Test base64 upload to the API (alternative method for Java backend)

    Args:
        image_path: Path to the image file
        api_url: API endpoint URL
    """
    print(f"\n{'='*60}")
    print(f"Testing Base64 Upload: {image_path}")
    print(f"API URL: {api_url}")
    print(f"{'='*60}")

    if not os.path.exists(image_path):
        print(f"❌ ERROR: File not found: {image_path}")
        return False

    try:
        import base64

        # Read and encode file as base64 (like Java would do)
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        print(f"📤 Sending base64 encoded file")
        print(f"   Original size: {len(image_bytes)} bytes")
        print(f"   Base64 size: {len(image_base64)} chars")

        # Send as JSON
        payload = {
            'image': image_base64
        }

        response = requests.post(api_url, json=payload)

        print(f"\n📥 Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"\n📊 Results:")
            print(f"   Prediction: {result.get('prediction', 'N/A')}")
            print(f"   Has Tumor: {result.get('has_tumor', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 0)*100:.2f}%")

            if result.get('has_tumor'):
                print(f"\n🔬 Tumor Type Analysis:")
                print(f"   Type: {result.get('tumor_type', 'N/A')}")
                print(f"   Confidence: {result.get('tumor_type_confidence', 0)*100:.2f}%")

            return True
        else:
            print(f"❌ ERROR: Request failed")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_check(api_url='http://localhost:5000/health'):
    """Test health check endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing Health Check")
    print(f"API URL: {api_url}")
    print(f"{'='*60}")

    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Server is healthy")
            print(f"   Binary model loaded: {result.get('binary_model_loaded', False)}")
            print(f"   Hybrid model loaded: {result.get('hybrid_model_loaded', False)}")
            print(f"   Device: {result.get('device', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Cannot connect to server")
        print(f"   Make sure the Flask server is running: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("Java Backend Compatibility Test Suite")
    print("="*60)

    # Check if server is running
    if not test_health_check():
        print("\n❌ Server is not running. Please start it with:")
        print("   python api_server.py")
        return 1

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

    if test_image is None:
        print("\n⚠️  No test images found. Please specify an image path:")
        print("   Example: python test_java_backend.py path/to/image.jpg")

        if len(sys.argv) > 1:
            test_image = sys.argv[1]
            if not os.path.exists(test_image):
                print(f"❌ Image not found: {test_image}")
                return 1
        else:
            return 1

    results = []

    # Test 1: File upload (multipart/form-data)
    print("\n" + "="*60)
    print("Test 1: Multipart File Upload (Java standard)")
    print("="*60)
    results.append(("Multipart Upload", test_file_upload(test_image)))

    # Test 2: Base64 upload (JSON)
    print("\n" + "="*60)
    print("Test 2: Base64 JSON Upload (Java alternative)")
    print("="*60)
    results.append(("Base64 Upload", test_base64_upload(test_image)))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✅ All tests passed! Java backend integration is working.")
        print("\n💡 Java Backend Usage Examples:")
        print("\n1. Multipart/Form-Data Upload (Recommended):")
        print("""
// Java Example with OkHttp
RequestBody requestBody = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("file", fileName,
        RequestBody.create(fileBytes, MediaType.parse("image/jpeg")))
    .build();

Request request = new Request.Builder()
    .url("http://localhost:5000/api/predict")
    .post(requestBody)
    .build();
        """)

        print("\n2. Base64 JSON Upload:")
        print("""
// Java Example with Base64
String base64Image = Base64.getEncoder().encodeToString(imageBytes);
String json = "{\\"image\\": \\"" + base64Image + "\\"}";

RequestBody body = RequestBody.create(json, MediaType.parse("application/json"));
Request request = new Request.Builder()
    .url("http://localhost:5000/api/predict")
    .post(body)
    .build();
        """)
        return 0
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

