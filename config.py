import os
from base64 import b64decode
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for mental health chatbot settings."""
    # Server settings
    PORT = int(os.getenv("PORT", 10000))
    
    # Security settings
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", 'gKrIjy-esAkcFlwKR3z73gsCcxWOSaRMQzrHDkCVOL0=')
    SESSION_EXPIRY_MINUTES = int(os.getenv("SESSION_EXPIRY_MINUTES", 30))
    TOKEN_EXPIRY_HOURS = int(os.getenv("TOKEN_EXPIRY_HOURS", 24))
    SECURE_COOKIES = os.getenv("SECURE_COOKIES", "FALSE").upper() == "TRUE"
    
    # Conversation settings
    MAX_CONVERSATION_LENGTH = int(os.getenv("MAX_CONVERSATION_LENGTH", 100))
    CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", 10))
    
    # API Keys
    HF_API_KEY = os.getenv("HF_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Resource links
    RESOURCE_LINKS = {
        'general': 'https://www.nimh.nih.gov',
        'crisis': 'https://988lifeline.org',
        'sleep': 'https://www.nimh.nih.gov/health/topics/sleep-disorders',
        'anxiety': 'https://www.nimh.nih.gov/health/topics/anxiety-disorders',
        'depression': 'https://www.nimh.nih.gov/health/topics/depression',
        'grief': 'https://www.nimh.nih.gov/health/topics/grief-and-loss',
        'trauma': 'https://www.nimh.nih.gov/health/topics/coping-with-traumatic-events',
        'support_groups': 'https://www.nami.org/Support-Education/Support-Groups',
        'therapy': 'https://www.psychologytoday.com/us/therapists'
    }
    
    # Mental health domain constraints
    MENTAL_HEALTH_TOPICS = [
        # Conditions and disorders
        "depression", "anxiety", "stress", "grief", "trauma", "ptsd", "ocd",
        "bipolar", "schizophrenia", "adhd", "add", "eating disorder", "anorexia", 
        "bulimia", "binge eating", "panic attack", "phobia", "insomnia",
        
        # Approaches and treatments
        "therapy", "counseling", "psychiatry", "psychology", "mental health",
        "coping", "mindfulness", "meditation", "self-care", "support group",
        "cognitive behavioral", "cbt", "dbt", "psychotherapy", "treatment",
        
        # Emotional states
        "emotion", "feeling", "mood", "sadness", "happiness", "anger", "fear",
        "loneliness", "isolation", "burnout", "exhaustion", "overwhelm",
        
        # Related concepts
        "wellbeing", "wellness", "mental wellness", "emotional health",
        "resilience", "recovery", "healing", "self-esteem", "confidence",
        "boundaries", "relationship", "social anxiety"
    ]
    
    # Crisis keywords for escalation
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "harm myself", "end my life", "want to die",
        "don't want to live", "no reason to live", "emergency", "crisis"
    ]
    
    # API fallback preferences
    PREFER_GEMINI = os.getenv("PREFER_GEMINI", "TRUE").upper() == "TRUE"
    USE_OPENAI_FALLBACK = os.getenv("USE_OPENAI_FALLBACK", "TRUE").upper() == "TRUE"
    
    # User interface settings
    DEFAULT_THEME = "light"
    AVAILABLE_THEMES = ["light", "dark", "system"]
    
    # User settings
    DEFAULT_RESPONSE_STYLE = "neutral"
    AVAILABLE_RESPONSE_STYLES = ["neutral", "friendly", "professional"]