"""
Application Initialization Script
Sets up NLTK data paths and validates ML components before app startup.
"""

import os
import sys
import logging
import nltk
from pathlib import Path

def initialize_nltk():
    """Initialize NLTK with proper data paths"""
    
    # Set NLTK data directory
    nltk_data_dir = os.environ.get('NLTK_DATA', '/opt/render/project/src/nltk_data')
    
    # Add custom NLTK data path
    if nltk_data_dir and os.path.exists(nltk_data_dir):
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
        logging.info(f"Added NLTK data path: {nltk_data_dir}")
    
    # Verify NLTK data is accessible
    try:
        from nltk.tokenize import word_tokenize
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # Test tokenization
        test_tokens = word_tokenize("Hello world")
        logging.info(f"NLTK tokenization test: {len(test_tokens)} tokens")
        
        # Test sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        test_sentiment = sia.polarity_scores("This is a test")
        logging.info(f"NLTK sentiment test: {test_sentiment}")
        
        return True
        
    except Exception as e:
        logging.error(f"NLTK initialization failed: {e}")
        return False

def initialize_spacy():
    """Initialize SpaCy model"""
    try:
        import spacy
        
        # Try to load the English model
        nlp = spacy.load("en_core_web_sm")
        
        # Test the model
        doc = nlp("This is a test sentence.")
        logging.info(f"SpaCy model test: {len(doc)} tokens, {len(list(doc.ents))} entities")
        
        return True
        
    except Exception as e:
        logging.error(f"SpaCy initialization failed: {e}")
        return False

def initialize_pytorch():
    """Initialize PyTorch with optimal settings"""
    try:
        import torch
        
        # Set optimal settings for CPU deployment
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        # Test basic operations
        test_tensor = torch.randn(2, 3)
        logging.info(f"PyTorch test: tensor shape {test_tensor.shape}")
        
        # Log device information
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"PyTorch device: {device}")
        
        return True
        
    except Exception as e:
        logging.error(f"PyTorch initialization failed: {e}")
        return False

def initialize_transformers():
    """Initialize Transformers library"""
    try:
        from transformers import pipeline
        
        # Set environment variables for optimal performance
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Set cache directory to a writable location
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/tmp/transformers_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        logging.info(f"Transformers cache directory: {cache_dir}")
        
        return True
        
    except Exception as e:
        logging.error(f"Transformers initialization failed: {e}")
        return False

def validate_ml_stack():
    """Validate all ML components"""
    components = {
        'NLTK': initialize_nltk,
        'SpaCy': initialize_spacy,
        'PyTorch': initialize_pytorch,
        'Transformers': initialize_transformers
    }
    
    results = {}
    all_passed = True
    
    for name, init_func in components.items():
        try:
            success = init_func()
            results[name] = 'OK' if success else 'FAILED'
            if not success:
                all_passed = False
        except Exception as e:
            logging.error(f"{name} validation failed: {e}")
            results[name] = 'ERROR'
            all_passed = False
    
    # Log results
    logging.info("ML Stack Validation Results:")
    for component, status in results.items():
        logging.info(f"  {component}: {status}")
    
    return all_passed, results

def initialize_app():
    """Main initialization function"""
    logging.info("🚀 Initializing Mental Health Chatbot...")
    
    # Validate ML stack
    success, results = validate_ml_stack()
    
    if success:
        logging.info("✅ All ML components initialized successfully!")
        return True
    else:
        logging.warning("⚠️  Some ML components failed to initialize")
        logging.info("📋 Component status:")
        for component, status in results.items():
            logging.info(f"   {component}: {status}")
        
        # Check if we have at least basic functionality
        critical_components = ['NLTK', 'PyTorch']
        critical_ok = all(results.get(comp) == 'OK' for comp in critical_components)
        
        if critical_ok:
            logging.info("✅ Critical components OK, proceeding with degraded functionality")
            return True
        else:
            logging.error("❌ Critical components failed, cannot start application")
            return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = initialize_app()
    sys.exit(0 if success else 1)