import re
import logging

class SafetyChecker:
    def __init__(self):
        # Patterns that warrant flagging but allow handling within the mental health context
        self.mental_health_patterns = [
            r'\b(suicide|self-harm|self harm|kill myself|killing myself)\b',
            r'\b(hurt myself|harming myself)\b'
        ]
        
        # Patterns that are entirely inappropriate for the chatbot to engage with
        self.unsafe_patterns = [
            r'\b(bomb|terrorist|attack plan|weapon|massacre|shooting)\b',
            r'\b(child porn|cp|csam|pedophil|underage|child abuse)\b',
            r'\b(rape|sexual assault)\b',
            r'\b(kill|murder|harm|attack) (others|people|someone|him|her|them)\b',
            r'\b(hack|ddos|phish|malware|ransomware)\b',
            r'\b(illegal drug|cocaine|heroin|meth production|drug dealing)\b'
        ]
        
        # Patterns for detecting inappropriate requests outside mental health scope
        self.inappropriate_request_patterns = [
            r'\b(dating advice|pickup|get (girl|guy|women|men))\b',
            r'\b(write|generate) (my|an) (essay|assignment|homework)\b',
            r'\b(create|write) (a|an) (advertisement|marketing)\b',
            r'\b(how to|ways to) (cheat|plagiarize|steal)\b',
            r'\b(stock|crypto|investment) (tips|advice|recommendation)\b'
        ]

    def is_safe(self, text: str) -> bool:
        """
        Determines if the input is safe to process.
        Mental health concerns like suicidal thoughts are "safe" in that they should be addressed
        rather than rejected, but with appropriate crisis resources.
        """
        try:
            text_lower = text.lower()
            
            # Check for entirely unsafe content first
            for pattern in self.unsafe_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    logging.warning(f"Unsafe content detected in text: {text} (matched pattern: {pattern})")
                    return False
            
            # Check for inappropriate requests outside mental health scope
            for pattern in self.inappropriate_request_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    logging.warning(f"Inappropriate request detected in text: {text} (matched pattern: {pattern})")
                    return False
            
            # Mental health concerns are "safe" - they should be handled appropriately rather than rejected
            for pattern in self.mental_health_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    logging.info(f"Mental health concern detected in text: {text} (matched pattern: {pattern})")
                    # We return True here because we want to address these concerns, not block them
                    return True
            
            return True
        except Exception as e:
            logging.error(f"Safety check error: {str(e)}")
            return False