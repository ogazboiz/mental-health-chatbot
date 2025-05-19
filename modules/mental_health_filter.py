import re
import logging
from config import Config

class MentalHealthFilter:
    def __init__(self):
        self.mental_health_topics = Config.MENTAL_HEALTH_TOPICS
        self.crisis_keywords = Config.CRISIS_KEYWORDS
        
        self.non_mental_health_patterns = [
            r'\b(stock market|investment|cryptocurrency|bitcoin|finance|trading)\b',
            r'\b(sports|football|basketball|baseball|soccer|game|match|score)\b',
            r'\b(politics|election|government|policy|politician|vote|campaign)\b',
            r'\b(recipe|cooking|baking|ingredients|dinner|lunch|breakfast)\b',
            r'\b(movie|film|tv show|television|actor|actress|director|watch)\b',
            r'\b(weather|forecast|temperature|rain|snow|sunny|cloudy|storm)\b',
            r'\b(travel|vacation|flight|hotel|tourist|destination|trip)\b',
            r'\b(news|headline|article|journalism|reporter|media)\b',
            r'\b(tech|technology|gadget|device|computer|software|hardware)\b',
            r'\b(shopping|product|buy|purchase|store|mall|online shop)\b'
        ]
        
    def is_mental_health_related(self, text: str) -> bool:
        """Determine if the input is related to mental health"""
        text_lower = text.lower()
        
        # Check for explicit mental health topics
        for topic in self.mental_health_topics:
            if topic in text_lower:
                logging.debug(f"Mental health topic detected: {topic}")
                return True
                
        # Check for questions about feelings or wellbeing (common mental health queries)
        wellbeing_patterns = [
            r'how (can|do) (i|you) (cope|deal|manage|handle)',
            r'(i\'m|i am|im) (feeling|so) (sad|down|anxious|depressed|worried|stressed)',
            r'(help|advice) (with|for) (my|dealing with|coping with)',
            r'(feel|feeling) (better|worse|good|bad|low|high)',
            r'having (trouble|difficulty|problems) with',
            r'(cant|can\'t|cannot) (stop|help) (thinking|feeling|worrying)'
        ]
        
        for pattern in wellbeing_patterns:
            if re.search(pattern, text_lower):
                logging.debug(f"Wellbeing pattern detected: {pattern}")
                return True
        
        # Check for excluded topics
        for pattern in self.non_mental_health_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logging.info(f"Non-mental health topic detected: {pattern}")
                return False
                
        # For ambiguous queries, default to accepting them
        # This is safer to avoid rejecting legitimate mental health concerns
        return True
    
    def contains_crisis_language(self, text: str) -> bool:
        """Check if the text contains crisis indicators"""
        text_lower = text.lower()
        
        # Direct keywords check
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                logging.warning(f"Crisis keyword detected: {keyword}")
                return True
        
        # More nuanced pattern matching for crisis indicators
        crisis_patterns = [
            r'(want|thinking about|considering) (to)? (die|suicide|kill myself|end it all)',
            r'(don\'t|do not) (want to|wanna) (live|be alive|exist) (anymore|any longer)',
            r'(no|zero) (point|reason|purpose) (in|to|for) (living|life|going on)',
            r'(everyone|world) (better|would be better) (off)? without me',
            r'(can\'t|cannot) (take|handle|deal with) (it|this) (anymore|any longer)',
            r'(plan|planning|preparing) (to|on) (hurt|harm|kill) (myself|me)',
            r'(this is|that\'s) (it|the end|my last|goodbye|farewell)'
        ]
        
        for pattern in crisis_patterns:
            if re.search(pattern, text_lower):
                logging.warning(f"Crisis pattern detected: {pattern}")
                return True
                
        return False
        
    def get_redirection_message(self, text: str) -> str:
        """Get a message to redirect non-mental health topics"""
        return ("I'm specialized in providing support for mental health concerns. "
                "While I can't help with that specific topic, I'm here if you'd like to "
                "discuss anything related to emotional wellbeing, stress, anxiety, or other mental health topics. "
                "Is there something about your mental or emotional wellbeing you'd like to talk about?")
                
    def get_crisis_resources(self) -> str:
        """Return crisis resources message"""
        return ("I'm concerned about your wellbeing. If you're in crisis or having thoughts of suicide, "
                "please reach out for immediate help:\n"
                "• Call or text 988 to reach the Suicide & Crisis Lifeline\n"
                "• Text HOME to 741741 to reach the Crisis Text Line\n"
                "• Call 911 or go to your nearest emergency room\n\n"
                "These services are free, confidential, and available 24/7. "
                "You deserve support, and help is available.")