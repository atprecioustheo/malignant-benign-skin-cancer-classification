"""
API Testing Script
Test your melanoma detection API endpoints
"""
import requests
import json
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change to your deployed URL
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "testpassword123"

class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
        
    def test_health(self):
        """Test health endpoint"""
        print("🏥 Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_models_endpoint(self):
        """Test models endpoint"""
        print("\n🤖 Testing models endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Models endpoint: {data}")
                return True
            else:
                print(f"❌ Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Models endpoint error: {e}")
            return False
    
    def test_register(self, email: str, password: str):
        """Test user registration"""
        print(f"\n👤 Testing registration for {email}...")
        
        try:
            response = self.session.post(
                f"{self.base_url}/auth/register",
                json={"email": email, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                print(f"✅ Registration successful: {data['user']['email']}")
                return True
            elif response.status_code == 400:
                print("⚠️ User already exists, trying login...")
                return self.test_login(email, password)
            else:
                print(f"❌ Registration failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False
    
    def test_login(self, email: str, password: str):
        """Test user login"""
        print(f"\n🔐 Testing login for {email}...")
        
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"email": email, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                print(f"✅ Login successful: {data['user']['email']}")
                return True
            else:
                print(f"❌ Login failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def set_auth_header(self):
        """Set authorization header"""
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            return True
        return False
    
    def test_dashboard(self):
        """Test dashboard endpoint"""
        print("\n📊 Testing dashboard endpoint...")
        
        if not self.set_auth_header():
            print("❌ No auth token available")
            return False
        
        try:
            response = self.session.get(f"{self.base_url}/dashboard")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Dashboard: {data}")
                return True
            else:
                print(f"❌ Dashboard failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
            return False
    
    def test_history(self):
        """Test history endpoint"""
        print("\n📝 Testing history endpoint...")
        
        if not self.set_auth_header():
            print("❌ No auth token available")
            return False
        
        try:
            response = self.session.get(f"{self.base_url}/history")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ History: Found {len(data)} records")
                return True
            else:
                print(f"❌ History failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ History error: {e}")
            return False
    
    def test_prediction(self, image_path: str = None):
        """Test prediction endpoint"""
        print("\n🔮 Testing prediction endpoint...")
        
        if not self.set_auth_header():
            print("❌ No auth token available")
            return False
        
        # Create a dummy image if none provided
        if image_path is None or not os.path.exists(image_path):
            print("Creating dummy test image...")
            from PIL import Image
            import numpy as np
            
            # Create a random image
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            image_path = "test_image.png"
            dummy_image.save(image_path)
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': ('test.png', f, 'image/png')}
                data = {
                    'name': 'Test Image',
                    'notes': 'API test image'
                }
                
                response = self.session.post(
                    f"{self.base_url}/predict",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction successful: {result}")
                return True
            else:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False
        finally:
            # Clean up dummy image
            if image_path == "test_image.png" and os.path.exists(image_path):
                os.remove(image_path)
    
    def run_all_tests(self, email: str, password: str, image_path: str = None):
        """Run all API tests"""
        print("🚀 Starting API tests...\n")
        
        results = {
            'health': self.test_health(),
            'models': self.test_models_endpoint(),
            'auth': self.test_register(email, password),
            'dashboard': False,
            'history': False,
            'prediction': False
        }
        
        if results['auth']:
            results['dashboard'] = self.test_dashboard()
            results['history'] = self.test_history()
            results['prediction'] = self.test_prediction(image_path)
        
        # Print summary
        print("\n" + "="*50)
        print("📋 TEST SUMMARY")
        print("="*50)
        
        passed = sum(results.values())
        total = len(results)
        
        for test, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test.upper():12} {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Your API is ready for deployment.")
        else:
            print("⚠️ Some tests failed. Please check the API configuration.")
        
        return results

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Melanoma Detection API')
    parser.add_argument('--url', default=API_BASE_URL, help='API base URL')
    parser.add_argument('--email', default=TEST_EMAIL, help='Test email')
    parser.add_argument('--password', default=TEST_PASSWORD, help='Test password')
    parser.add_argument('--image', help='Path to test image')
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    results = tester.run_all_tests(args.email, args.password, args.image)
    
    return results

if __name__ == "__main__":
    main()