#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

class BatiConnectAPITester:
    def __init__(self, base_url: str = "https://travy.preview.emergentagent.com"):
        self.base_url = base_url
        self.token = None
        self.user_data = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED")
        else:
            print(f"❌ {name}: FAILED - {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details,
            "response_data": response_data
        })

    def make_request(self, method: str, endpoint: str, data: Dict = None, expected_status: int = 200) -> tuple[bool, Dict]:
        """Make API request and return success status and response data"""
        url = f"{self.base_url}/api/{endpoint.lstrip('/')}"
        headers = {'Content-Type': 'application/json'}
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)
            elif method.upper() == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=10)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                return False, {"error": f"Unsupported method: {method}"}

            success = response.status_code == expected_status
            try:
                response_data = response.json()
            except:
                response_data = {"status_code": response.status_code, "text": response.text}

            return success, response_data

        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

    def test_root_endpoint(self):
        """Test API root endpoint"""
        success, response = self.make_request('GET', '/')
        self.log_test(
            "API Root Endpoint", 
            success and "BâtiConnect API" in str(response),
            f"Response: {response}" if not success else "",
            response
        )
        return success

    def test_stats_endpoint(self):
        """Test stats endpoint"""
        success, response = self.make_request('GET', '/stats')
        expected_keys = ['total_professionals', 'verified_professionals', 'total_reviews', 'total_users']
        
        if success:
            has_all_keys = all(key in response for key in expected_keys)
            success = has_all_keys
            details = "" if has_all_keys else f"Missing keys: {[k for k in expected_keys if k not in response]}"
        else:
            details = f"Request failed: {response}"

        self.log_test("Stats Endpoint", success, details, response)
        return success

    def test_professionals_endpoint(self):
        """Test professionals list endpoint"""
        success, response = self.make_request('GET', '/professionals')
        
        if success:
            is_list = isinstance(response, list)
            success = is_list
            details = "" if is_list else f"Expected list, got: {type(response)}"
        else:
            details = f"Request failed: {response}"

        self.log_test("Professionals List Endpoint", success, details, response)
        return success

    def test_user_registration(self):
        """Test user registration"""
        timestamp = int(datetime.now().timestamp())
        test_user = {
            "email": f"test.user.{timestamp}@example.com",
            "password": "TestPass123!",
            "name": f"Test User {timestamp}",
            "user_type": "particulier",
            "phone": "0612345678",
            "city": "Paris",
            "postal_code": "75001"
        }

        success, response = self.make_request('POST', '/auth/register', test_user, 200)
        
        if success:
            has_token = 'access_token' in response
            has_user = 'user' in response
            success = has_token and has_user
            
            if success:
                self.token = response['access_token']
                self.user_data = response['user']
                details = f"User created: {self.user_data.get('email')}"
            else:
                details = f"Missing token or user data: {response}"
        else:
            details = f"Registration failed: {response}"

        self.log_test("User Registration", success, details, response)
        return success

    def test_user_login(self):
        """Test user login with created user"""
        if not self.user_data:
            self.log_test("User Login", False, "No user data available for login test")
            return False

        login_data = {
            "email": self.user_data['email'],
            "password": "TestPass123!"
        }

        success, response = self.make_request('POST', '/auth/login', login_data, 200)
        
        if success:
            has_token = 'access_token' in response
            has_user = 'user' in response
            success = has_token and has_user
            details = f"Login successful for: {login_data['email']}" if success else f"Missing token or user: {response}"
        else:
            details = f"Login failed: {response}"

        self.log_test("User Login", success, details, response)
        return success

    def test_auth_me_endpoint(self):
        """Test /auth/me endpoint with token"""
        if not self.token:
            self.log_test("Auth Me Endpoint", False, "No token available")
            return False

        success, response = self.make_request('GET', '/auth/me')
        
        if success:
            has_user_id = 'user_id' in response
            has_email = 'email' in response
            success = has_user_id and has_email
            details = f"User data retrieved: {response.get('email')}" if success else f"Missing user fields: {response}"
        else:
            details = f"Auth me failed: {response}"

        self.log_test("Auth Me Endpoint", success, details, response)
        return success

    def test_professional_registration(self):
        """Test professional registration with SIRET"""
        timestamp = int(datetime.now().timestamp())
        pro_user = {
            "email": f"pro.test.{timestamp}@example.com",
            "password": "TestPass123!",
            "name": f"Pro Test {timestamp}",
            "user_type": "professionnel",
            "siret": "12345678900123",  # Valid format SIRET
            "profession": "Plombier",
            "phone": "0612345679",
            "city": "Lyon",
            "postal_code": "69001",
            "description": "Plombier professionnel avec 10 ans d'expérience",
            "years_experience": 10
        }

        success, response = self.make_request('POST', '/auth/register', pro_user, 200)
        
        if success:
            has_token = 'access_token' in response
            has_user = 'user' in response
            is_pro = response.get('user', {}).get('user_type') == 'professionnel'
            success = has_token and has_user and is_pro
            details = f"Professional created: {response.get('user', {}).get('email')}" if success else f"Invalid pro registration: {response}"
        else:
            details = f"Pro registration failed: {response}"

        self.log_test("Professional Registration", success, details, response)
        return success

    def test_contact_form_submission(self):
        """Test contact form submission"""
        if not self.user_data:
            self.log_test("Contact Form Submission", False, "No user data available")
            return False

        # First get a professional to contact
        success, pros_response = self.make_request('GET', '/professionals')
        if not success or not pros_response:
            self.log_test("Contact Form Submission", False, "No professionals available to contact")
            return False

        if not isinstance(pros_response, list) or len(pros_response) == 0:
            self.log_test("Contact Form Submission", False, "No professionals in response")
            return False

        pro_id = pros_response[0].get('user_id')
        if not pro_id:
            self.log_test("Contact Form Submission", False, "Professional has no user_id")
            return False

        contact_data = {
            "professional_id": pro_id,
            "name": "Test Client",
            "email": "test.client@example.com",
            "phone": "0612345680",
            "message": "Test message for contact form",
            "project_type": "Rénovation"
        }

        success, response = self.make_request('POST', '/contact', contact_data, 200)
        
        if success:
            has_contact_id = 'contact_id' in response
            success = has_contact_id
            details = f"Contact form submitted: {response.get('contact_id')}" if success else f"Missing contact_id: {response}"
        else:
            details = f"Contact submission failed: {response}"

        self.log_test("Contact Form Submission", success, details, response)
        return success

    def test_professions_endpoint(self):
        """Test professions list endpoint"""
        success, response = self.make_request('GET', '/professions')
        
        if success:
            is_list = isinstance(response, list)
            success = is_list
            details = f"Found {len(response)} professions" if is_list else f"Expected list, got: {type(response)}"
        else:
            details = f"Request failed: {response}"

        self.log_test("Professions List Endpoint", success, details, response)
        return success

    def run_all_tests(self):
        """Run all API tests"""
        print(f"🚀 Starting BâtiConnect API Tests")
        print(f"📍 Base URL: {self.base_url}")
        print("=" * 60)

        # Basic API tests
        self.test_root_endpoint()
        self.test_stats_endpoint()
        self.test_professionals_endpoint()
        self.test_professions_endpoint()

        # Authentication tests
        self.test_user_registration()
        self.test_user_login()
        self.test_auth_me_endpoint()

        # Professional registration
        self.test_professional_registration()

        # Contact form
        self.test_contact_form_submission()

        print("=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed!")
            failed_tests = [r for r in self.test_results if not r['success']]
            print("\nFailed tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")
            return 1

def main():
    """Main test runner"""
    tester = BatiConnectAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())