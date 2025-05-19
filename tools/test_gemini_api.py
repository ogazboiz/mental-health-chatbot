#!/usr/bin/env python3
"""
Gemini Direct API Test

Tests the Gemini API using the direct REST API call to match the successful curl command.
"""

import os
import json
import sys
import httpx
from dotenv import load_dotenv

def test_gemini_direct_api():
    """Test direct Gemini API call using same format as successful curl command"""
    # Load environment variables
    load_dotenv()
    
    # Get the API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: GEMINI_API_KEY not found in environment variables")
        print("Please add your Gemini API key to your .env file.\n")
        return False
    
    # Use the exact same model and endpoint as the curl command
    model = "gemini-2.0-flash"
    api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    print(f"\n🔑 API Key found: {api_key[:4]}...{api_key[-4:]}")
    print(f"🤖 Model: {model}")
    print(f"🌐 API Endpoint: {api_endpoint}")
    print("Making direct API call...")
    
    # Create the exact same payload as the curl command
    payload = {
        "contents": [{
            "parts": [{"text": "Say hello and explain what mental health is in one sentence"}]
        }]
    }
    
    # Make the API call
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                api_endpoint,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code != 200:
                print(f"\n❌ ERROR: API call failed with status code {response.status_code}")
                print(f"Response: {response.text}\n")
                return False
            
            # Parse the response
            result = response.json()
            
            # Extract the text from the response
            if (
                "candidates" in result 
                and len(result["candidates"]) > 0 
                and "content" in result["candidates"][0]
                and "parts" in result["candidates"][0]["content"]
                and len(result["candidates"][0]["content"]["parts"]) > 0
                and "text" in result["candidates"][0]["content"]["parts"][0]
            ):
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                
                print("\n✅ SUCCESS: API call successful!")
                print(f"Response: {text[:100]}...\n")
                
                # Print the full response structure for debugging
                print("Full Response Structure:")
                print(json.dumps(result, indent=2)[:300] + "...\n")
                
                print("\nImplementation Instructions:")
                print("1. Replace modules/mental_health_response_generator.py with the provided file")
                print("2. This implementation uses your exact API key and endpoint format")
                print("3. It implements a cascading fallback system:")
                print("   a. Direct Gemini API call (like your curl command)")
                print("   b. Gemini library method (as backup)")
                print("   c. OpenAI fallback (if configured)")
                print("   d. Built-in responses (final fallback)")
                
                return True
            else:
                print("\n❌ ERROR: Could not extract text from response")
                print(f"Response structure: {json.dumps(result, indent=2)}\n")
                return False
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        return False

if __name__ == "__main__":
    print("\n🧪 Testing Gemini Direct API")
    print("---------------------------")
    success = test_gemini_direct_api()
    sys.exit(0 if success else 1)