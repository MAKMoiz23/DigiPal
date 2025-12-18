"""
Test script to verify Google Gemini API integration
Run this after setting up your .env file with GOOGLE_API_KEY
"""

import os
import sys
from dotenv import load_dotenv

def test_env_setup():
    """Test if environment variables are properly configured"""
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        return False
    
    if api_key == "your_google_api_key_here":
        print("❌ Please update GOOGLE_API_KEY in .env file with your actual API key")
        return False
    
    print("✅ Environment variables are properly configured")
    return True

def test_imports():
    """Test if required packages can be imported"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        print("✅ langchain-google-genai imports successful")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run: pip install langchain-google-genai python-dotenv")
        return False

def test_gemini_connection():
    """Test basic connection to Gemini API"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        model = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL_NAME"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        
        # Simple test query
        response = model.invoke("Hello, can you respond with 'DigiPal is ready!'?")
        print(f"✅ Gemini API connection successful: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing DigiPal Google Gemini Integration...")
    print("=" * 50)
    
    # Test environment setup
    if not test_env_setup():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test API connection
    if not test_gemini_connection():
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 All tests passed! DigiPal is ready to use with Google Gemini API")
    print("\nNext steps:")
    print("1. Run: python chat.py (for command line interface)")
    print("2. Run: python UI.py (for web interface)")