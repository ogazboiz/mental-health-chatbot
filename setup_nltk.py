#!/usr/bin/env python3
"""
NLTK Data Setup Script for Render Deployment
This script sets up NLTK data using local files and downloads when needed.
"""

import os
import sys
import zipfile
import nltk
from pathlib import Path

def setup_nltk_data():
    """Set up NLTK data for the application"""
    
    # Set NLTK data path
    nltk_data_dir = os.environ.get('NLTK_DATA', '/opt/render/project/src/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add to NLTK data path
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    
    print(f"NLTK data directory: {nltk_data_dir}")
    
    # Setup punkt tokenizer - try local zip file first
    punkt_zip_path = "punkt.zip"
    if os.path.exists(punkt_zip_path):
        try:
            tokenizers_dir = os.path.join(nltk_data_dir, "tokenizers")
            os.makedirs(tokenizers_dir, exist_ok=True)
            
            print(f"Extracting {punkt_zip_path} to {tokenizers_dir}")
            with zipfile.ZipFile(punkt_zip_path, 'r') as zip_ref:
                zip_ref.extractall(tokenizers_dir)
            
            print("✅ Punkt tokenizer extracted successfully")
            
        except Exception as e:
            print(f"❌ Error extracting punkt.zip: {e}")
            print("📥 Downloading punkt from NLTK...")
            try:
                nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
                print("✅ Punkt downloaded successfully")
            except Exception as download_error:
                print(f"❌ Failed to download punkt: {download_error}")
                return False
    else:
        print("📥 punkt.zip not found, downloading from NLTK...")
        try:
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
            print("✅ Punkt downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download punkt: {e}")
            return False
    
    # Download other required NLTK data
    required_data = ['vader_lexicon', 'stopwords']
    for data_name in required_data:
        try:
            print(f"📥 Downloading {data_name}...")
            nltk.download(data_name, download_dir=nltk_data_dir, quiet=True)
            print(f"✅ {data_name} downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download {data_name}: {e}")
    
    # Verify NLTK data is accessible
    try:
        from nltk.tokenize import word_tokenize
        test_tokens = word_tokenize("This is a test sentence.")
        print(f"✅ NLTK tokenization test successful: {len(test_tokens)} tokens")
        return True
    except Exception as e:
        print(f"❌ NLTK tokenization test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up NLTK data...")
    success = setup_nltk_data()
    if success:
        print("✅ NLTK setup completed successfully!")
        sys.exit(0)
    else:
        print("❌ NLTK setup failed!")
        sys.exit(1)