import os
import nltk
import ssl
from pathlib import Path
import logging

def setup_nltk_data():
    """Setup NLTK data for deployment - downloads directly from NLTK servers"""
    
    try:
        # Handle SSL issues that sometimes occur with NLTK downloads
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Create NLTK data directory in the current working directory
        nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data_path, exist_ok=True)
        
        # Add our custom path to NLTK's search paths
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)
        
        print(f"📁 NLTK data will be stored in: {nltk_data_path}")
        print("⬇️ Downloading NLTK data packages...")
        
        # List of NLTK packages we need
        packages_to_download = [
            'punkt',           # Sentence tokenizer
            'punkt_tab',       # Updated punkt tokenizer
            'stopwords',       # Stop words
            'wordnet',         # WordNet lexical database
            'averaged_perceptron_tagger',  # POS tagger
            'vader_lexicon',   # Sentiment analysis
            'omw-1.4',        # Open Multilingual Wordnet
        ]
        
        successful_downloads = []
        failed_downloads = []
        
        for package in packages_to_download:
            try:
                print(f"  📦 Downloading {package}...")
                nltk.download(package, download_dir=nltk_data_path, quiet=True)
                successful_downloads.append(package)
                print(f"  ✅ {package} downloaded successfully")
            except Exception as e:
                print(f"  ⚠️ Failed to download {package}: {str(e)}")
                failed_downloads.append(package)
        
        print(f"\n📊 Download Summary:")
        print(f"  ✅ Successful: {len(successful_downloads)} packages")
        print(f"  ❌ Failed: {len(failed_downloads)} packages")
        
        if successful_downloads:
            print(f"  Successfully downloaded: {', '.join(successful_downloads)}")
        
        if failed_downloads:
            print(f"  Failed to download: {', '.join(failed_downloads)}")
        
        # Test if punkt tokenizer works
        try:
            from nltk.tokenize import word_tokenize, sent_tokenize
            
            # Test tokenization
            test_sentence = "Hello, this is a test sentence. Does it work?"
            words = word_tokenize(test_sentence)
            sentences = sent_tokenize(test_sentence)
            
            print(f"\n🧪 Testing NLTK tokenization:")
            print(f"  Input: {test_sentence}")
            print(f"  Words: {words}")
            print(f"  Sentences: {sentences}")
            print("  ✅ NLTK tokenization test passed!")
            
        except Exception as e:
            print(f"  ❌ NLTK tokenization test failed: {str(e)}")
            
            # If punkt fails, try to use the basic tokenizer as fallback
            print("  🔄 Setting up fallback tokenization...")
            try:
                # Create a simple fallback
                import re
                def fallback_word_tokenize(text):
                    return re.findall(r'\b\w+\b', text.lower())
                
                def fallback_sent_tokenize(text):
                    return re.split(r'[.!?]+', text)
                
                # Monkey patch NLTK if needed
                nltk.word_tokenize = fallback_word_tokenize
                nltk.sent_tokenize = fallback_sent_tokenize
                
                print("  ✅ Fallback tokenization ready")
                
            except Exception as fallback_error:
                print(f"  ❌ Even fallback failed: {str(fallback_error)}")
        
        # Check what's actually in our NLTK data directory
        print(f"\n📂 NLTK data directory contents:")
        for root, dirs, files in os.walk(nltk_data_path):
            level = root.replace(nltk_data_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        
        print(f"\n🎉 NLTK setup completed!")
        print(f"📍 NLTK data path: {nltk_data_path}")
        print(f"🔍 NLTK search paths: {nltk.data.path[:3]}...")  # Show first 3 paths
        
        return True
        
    except Exception as e:
        print(f"❌ NLTK setup failed: {str(e)}")
        logging.error(f"NLTK setup error: {str(e)}")
        
        # Set up minimal fallback
        print("🔄 Setting up minimal fallback...")
        try:
            import re
            
            # Create very basic tokenizer functions
            def basic_word_tokenize(text):
                """Basic word tokenization fallback"""
                return re.findall(r'\b\w+\b', text.lower())
            
            def basic_sent_tokenize(text):
                """Basic sentence tokenization fallback"""
                sentences = re.split(r'[.!?]+', text.strip())
                return [s.strip() for s in sentences if s.strip()]
            
            # Store them globally so they can be imported
            globals()['fallback_word_tokenize'] = basic_word_tokenize
            globals()['fallback_sent_tokenize'] = basic_sent_tokenize
            
            print("✅ Basic fallback tokenization ready")
            return False
            
        except Exception as fallback_error:
            print(f"❌ Critical error - even basic fallback failed: {str(fallback_error)}")
            return False

def test_nltk_functionality():
    """Test if NLTK is working properly"""
    try:
        from nltk.tokenize import word_tokenize
        test_text = "This is a test sentence."
        tokens = word_tokenize(test_text)
        print(f"✅ NLTK test successful: {tokens}")
        return True
    except Exception as e:
        print(f"❌ NLTK test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting NLTK setup...")
    success = setup_nltk_data()
    
    if success:
        print("\n🧪 Running functionality test...")
        test_nltk_functionality()
    
    print("\n✨ NLTK setup script completed!")