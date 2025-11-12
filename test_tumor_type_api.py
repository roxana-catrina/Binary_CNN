"""
Test script to verify the API returns tumor type when tumor is detected
"""
import requests
import json

def test_predict_endpoint():
    """Test the /api/predict endpoint with an image"""
    url = "http://localhost:5000/api/predict"

    # Test with a test image
    test_image_path = "my_test_images/brain_scan.jpg"

    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)

        print("Status Code:", response.status_code)
        print("\nResponse JSON:")
        result = response.json()
        print(json.dumps(result, indent=2))

        # Check if response contains tumor type when tumor is detected
        if result.get('has_tumor'):
            if 'tumor_type' in result:
                print("\n✓ Tumor type detected:", result['tumor_type'])
                print("✓ Tumor type confidence:", result.get('tumor_type_confidence'))
                print("✓ Tumor type probabilities:", result.get('tumor_type_probabilities'))
            else:
                print("\n✗ ERROR: Tumor detected but no tumor type returned!")
        else:
            print("\n✓ No tumor detected, tumor type not needed")

    except FileNotFoundError:
        print(f"Error: Test image not found at {test_image_path}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server. Make sure it's running on http://localhost:5000")
    except Exception as e:
        print(f"Error: {e}")

def test_health_endpoint():
    """Test the /health endpoint"""
    url = "http://localhost:5000/health"

    try:
        response = requests.get(url)
        print("Health Check Status Code:", response.status_code)
        print("Health Check Response:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server. Make sure it's running on http://localhost:5000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Brain Tumor API with Tumor Type Detection")
    print("=" * 60)

    print("\n1. Testing Health Endpoint...")
    print("-" * 60)
    test_health_endpoint()

    print("\n\n2. Testing Predict Endpoint...")
    print("-" * 60)
    test_predict_endpoint()

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

