"""
Test script for the Flask API
Run this after starting the api_server.py to verify it works
"""
import requests
import json
import os

# Configuration
API_URL = "http://localhost:5000"
TEST_IMAGE = "my_test_images/brain_scan.jpg"

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_multipart():
    """Test prediction with multipart file upload"""
    print("\n=== Testing Prediction (Multipart Upload) ===")

    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Test image not found: {TEST_IMAGE}")
        return False

    try:
        with open(TEST_IMAGE, 'rb') as f:
            files = {'file': ('brain_scan.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/api/predict", files=files)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        result = response.json()
        if result.get('success'):
            print(f"\n✅ Prediction: {result['prediction']}")
            print(f"✅ Has Tumor: {result['has_tumor']}")
            print(f"✅ Confidence: {result['confidence'] * 100:.2f}%")

            # Check for tumor type if tumor is detected
            if result.get('has_tumor'):
                if 'tumor_type' in result:
                    print(f"✅ Tumor Type: {result['tumor_type']}")
                    print(f"✅ Tumor Type Confidence: {result['tumor_type_confidence'] * 100:.2f}%")
                    print(f"✅ Tumor Type Probabilities:")
                    for label, prob in result.get('tumor_type_probabilities', {}).items():
                        print(f"   - {label}: {prob * 100:.2f}%")
                else:
                    print(f"⚠️ WARNING: Tumor detected but no tumor type provided!")

            return True
        else:
            print(f"\n❌ Prediction failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_base64():
    """Test prediction with base64 encoded image"""
    print("\n=== Testing Prediction (Base64 JSON) ===")

    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Test image not found: {TEST_IMAGE}")
        return False

    try:
        import base64

        with open(TEST_IMAGE, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            'image': image_data
        }

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(
            f"{API_URL}/api/predict",
            json=payload,
            headers=headers
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        result = response.json()
        if result.get('success'):
            print(f"\n✅ Prediction: {result['prediction']}")
            print(f"✅ Has Tumor: {result['has_tumor']}")
            print(f"✅ Confidence: {result['confidence'] * 100:.2f}%")
            return True
        else:
            print(f"\n❌ Prediction failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Brain Tumor Detection API Test")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"Test Image: {TEST_IMAGE}")

    # Run tests
    results = []

    results.append(("Health Check", test_health_check()))
    results.append(("Prediction (Multipart)", test_predict_multipart()))
    results.append(("Prediction (Base64)", test_predict_base64()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")

    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

