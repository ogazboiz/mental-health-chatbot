import os
import json
import asyncio
from typing import Dict, List, Any
import re
import logging
import httpx
from rake_nltk import Rake
from config import Config

class NLPProcessor:
    def __init__(self):
        self.neuroscience_terms = [
            "amygdala", "hippocampus", "prefrontal cortex", "limbic system", "cerebral cortex",
            "dopamine", "serotonin", "norepinephrine", "gaba", "glutamate", "depression", "anxiety"
        ]
        self.grief_keywords = ["lost", "loss", "died", "death", "grief", "bereavement", "passed", "gone"]
        self.emotional_keywords = ["sad", "anxious", "depressed", "down", "upset"]
        self.coping_keywords = ["cope", "coping", "ways", "strategies", "deal", "manage"]
        self.greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "good night"]
        self.rake = Rake()
        
        # Use the correct API key for Hugging Face
        self.hf_api_key = Config.HF_API_KEY
        
        # API endpoints
        self.hf_sentiment_url = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
        self.hf_emotion_url = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
        
        # Confidence thresholds for classification
        self.EMOTION_CONFIDENCE_THRESHOLD = 0.5
        self.SENTIMENT_CONFIDENCE_THRESHOLD = 0.4
        
        # MEMORY OPTIMIZATION: Remove heavy transformers models - use API-only approach
        # self.sentiment_classifier = None  # ← REMOVED to save memory
        # self.emotion_classifier = None    # ← REMOVED to save memory
        
        # Fallback rule-based analyzers
        self._init_rule_based_analyzers()
        
        # Test API key on initialization
        if self.hf_api_key:
            logging.info(f"NLP Processor initialized with HF API key: {self.hf_api_key[:10]}...")
            logging.info("MEMORY OPTIMIZED: Using API-only approach (no local models)")
        else:
            logging.warning("No Hugging Face API key found! Will use rule-based fallbacks only.")
        
        logging.info("NLP Processor initialized in memory-optimized mode")
        
    def _init_rule_based_analyzers(self):
        """Initialize rule-based analyzers for fallback"""
        # Additional emotion keywords for better rule-based detection
        self.sadness_keywords = ["sad", "down", "depressed", "hopeless", "miserable", "unhappy", "blue", "empty", "lonely"]
        self.anxiety_keywords = ["anxious", "nervous", "worried", "panicking", "panic", "scared", "afraid", "fear", "stress"]
        self.anger_keywords = ["angry", "mad", "frustrated", "irritated", "annoyed", "upset", "furious", "rage"]
        self.joy_keywords = ["happy", "joy", "excited", "glad", "pleased", "grateful", "thankful", "content"]
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        pass
        
    async def _query_hf_api_async(self, text: str, endpoint: str) -> List[Dict[str, Any]]:
        """Query Hugging Face inference API using httpx."""
        if not self.hf_api_key:
            raise Exception("No Hugging Face API key configured")
            
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            try:
                logging.debug(f"Making HF API request to {endpoint} with text: {text[:50]}...")
                response = await client.post(
                    endpoint,
                    json={"inputs": text},
                    headers=headers,
                    timeout=15.0
                )
                response.raise_for_status()
                result = response.json()
                logging.debug(f"HF API response received: {str(result)[:200]}...")
                return result
            except Exception as e:
                logging.error(f"Hugging Face API error for {endpoint}: {str(e)}")
                raise
    
    # MEMORY OPTIMIZATION: Removed _init_local_models method to save memory
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for intent, sentiment, emotions, and other features"""
        text_lower = text.lower()
        logging.debug(f"Analyzing text: {text}")
        
        # Intent classification (keyword-based)
        intent_map = {
            "LABEL_0": "greeting",
            "LABEL_1": "seeking_information", 
            "LABEL_2": "emotional_support",
            "LABEL_3": "coping_strategies",
            "LABEL_4": "resources_request",
            "LABEL_5": "personal_story",
            "LABEL_6": "crisis",
            "LABEL_7": "physical_symptom"
        }
        
        # Default intent
        intent = {"label": "LABEL_5", "confidence": 0.5, "intent": "general", "model_source": "keyword"}
        
        # Keyword-based intent detection
        if any(kw in text_lower for kw in self.greeting_keywords):
            intent = {"label": "LABEL_0", "confidence": 0.9, "intent": "greeting", "model_source": "keyword"}
        elif any(kw in text_lower for kw in self.coping_keywords):
            intent = {"label": "LABEL_3", "confidence": 0.9, "intent": "coping_strategies", "model_source": "keyword"}
        elif any(kw in text_lower for kw in self.emotional_keywords + self.grief_keywords):
            intent = {"label": "LABEL_2", "confidence": 0.9, "intent": "emotional_support", "model_source": "keyword"}
        elif "crisis" in text_lower or "urgent" in text_lower:
            intent = {"label": "LABEL_6", "confidence": 0.9, "intent": "crisis", "model_source": "keyword"}
        elif any(q in text_lower for q in ["what is", "how does", "why do", "when can", "where"]):
            intent = {"label": "LABEL_1", "confidence": 0.8, "intent": "seeking_information", "model_source": "keyword"}
        elif "resource" in text_lower or "referral" in text_lower or "help with" in text_lower:
            intent = {"label": "LABEL_4", "confidence": 0.8, "intent": "resources_request", "model_source": "keyword"}
        elif any(s in text_lower for s in ["symptom", "pain", "headache", "tired", "exhausted", "nauseous"]):
            intent = {"label": "LABEL_7", "confidence": 0.8, "intent": "physical_symptom", "model_source": "keyword"}
            
        # Sentiment classification - API first, then rule-based fallback
        sentiment = {"label": "neutral", "confidence": 0.5, "model_source": "default"}
        
        if self.hf_api_key:
            try:
                logging.info("Attempting Hugging Face sentiment analysis...")
                sentiment_result = await self._query_hf_api_async(text, self.hf_sentiment_url)
                
                if sentiment_result and isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                    items = sentiment_result[0] if isinstance(sentiment_result[0], list) else sentiment_result
                    
                    if items and isinstance(items, list):
                        top_sentiment = max(items, key=lambda x: x.get('score', 0))
                        score = top_sentiment.get('score', 0)
                        label = top_sentiment.get('label', 'neutral')
                        
                        if score < self.SENTIMENT_CONFIDENCE_THRESHOLD:
                            sentiment = {
                                "label": "neutral", 
                                "confidence": 0.5,
                                "raw_label": label,
                                "raw_score": score,
                                "model_source": "threshold_filter"
                            }
                        else:
                            sentiment = {
                                "label": "negative" if label in ["1 star", "2 stars"] else 
                                        "neutral" if label == "3 stars" else "positive",
                                "confidence": score,
                                "raw_label": label,
                                "model_source": "huggingface_api"
                            }
                        
            except Exception as e:
                logging.warning(f"HF API sentiment analysis failed: {str(e)}. Using rule-based fallback.")
                sentiment = self._rule_based_sentiment(text_lower)
        else:
            sentiment = self._rule_based_sentiment(text_lower)
        
        # Override with keyword detection
        if any(kw in text_lower for kw in self.grief_keywords):
            sentiment = {"label": "negative", "confidence": 0.9, "model_source": "keyword_grief"}
        elif any(kw in text_lower for kw in self.emotional_keywords):
            sentiment = {"label": "negative", "confidence": 0.85, "model_source": "keyword_emotional"}
            
        # Emotion classification - API first, then rule-based
        emotions = "none"
        emotion_source = "default"
        
        if self.hf_api_key:
            try:
                logging.info("Attempting Hugging Face emotion analysis...")
                emotion_results = await self._query_hf_api_async(text, self.hf_emotion_url)
                
                if emotion_results and isinstance(emotion_results, list) and len(emotion_results) > 0:
                    if len(emotion_results[0]) > 0:
                        top_emotion = max(emotion_results[0], key=lambda x: x.get('score', 0))
                        top_score = top_emotion.get('score', 0)
                        
                        if top_score > self.EMOTION_CONFIDENCE_THRESHOLD:
                            emotions = top_emotion.get('label', 'none').lower()
                            emotion_source = "huggingface_api"
                            logging.info(f"Emotion detected via HF API: {emotions} with score {top_score:.4f}")
                        else:
                            emotions = "none"
                            emotion_source = "threshold_filter"
                    
            except Exception as e:
                logging.warning(f"HF API emotion analysis failed: {str(e)}. Using rule-based fallback.")
                emotions = self._rule_based_emotion(text_lower)
                emotion_source = "rule_based"
        else:
            emotions = self._rule_based_emotion(text_lower)
            emotion_source = "rule_based"
        
        # Override with keyword detection
        if any(kw in text_lower for kw in self.grief_keywords):
            emotions = "grief"
            emotion_source = "keyword_grief"
        elif "sad" in text_lower:
            emotions = "sadness"
            emotion_source = "keyword_sad"
        elif any(term in text_lower for term in ["anxious", "nervous", "worry", "afraid", "scared"]):
            emotions = "fear"
            emotion_source = "keyword_anxiety"
        elif any(term in text_lower for term in ["angry", "mad", "frustrated", "annoyed"]):
            emotions = "anger"
            emotion_source = "keyword_anger"
            
        # Keyword extraction
        self.rake.extract_keywords_from_text(text)
        keywords = self.rake.get_ranked_phrases()[:5]
        
        # Neuroscience terms detection
        detected_terms = [term for term in self.neuroscience_terms 
                         if re.search(r'\b' + re.escape(term) + r'\b', text_lower)]
                         
        # Question detection
        is_question = text.strip().endswith('?') or any(text_lower.startswith(word) 
                     for word in ['what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'will'])
                     
        # Compile the analysis result
        result = {
            "intent": intent,
            "sentiment": sentiment,
            "emotions": emotions,
            "emotion_source": emotion_source,
            "neuroscience_terms": detected_terms,
            "keywords": keywords,
            "is_question": is_question,
            "entities": [],
            "processed_text": text,
            "is_neuroscience": bool(detected_terms),
            "is_response_to": None
        }
        
        logging.info(f"Analysis result: intent={intent['intent']}, sentiment={sentiment['label']}, emotion={emotions}")
        return result
        
    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Rule-based sentiment analysis as fallback"""
        negative_words = self.grief_keywords + self.sadness_keywords + self.anxiety_keywords + self.anger_keywords
        positive_words = self.joy_keywords
        
        negative_count = sum(1 for word in negative_words if word in text)
        positive_count = sum(1 for word in positive_words if word in text)
        
        if negative_count > positive_count:
            confidence = min(0.5 + (negative_count - positive_count) * 0.1, 0.9)
            return {"label": "negative", "confidence": confidence, "model_source": "rule_based"}
        elif positive_count > negative_count:
            confidence = min(0.5 + (positive_count - negative_count) * 0.1, 0.9)
            return {"label": "positive", "confidence": confidence, "model_source": "rule_based"}
        else:
            return {"label": "neutral", "confidence": 0.6, "model_source": "rule_based"}
    
    def _rule_based_emotion(self, text: str) -> str:
        """Rule-based emotion detection as fallback"""
        if any(kw in text for kw in self.grief_keywords):
            return "grief"
        
        emotion_counts = {
            "sadness": sum(1 for word in self.sadness_keywords if word in text),
            "fear": sum(1 for word in self.anxiety_keywords if word in text),
            "anger": sum(1 for word in self.anger_keywords if word in text),
            "joy": sum(1 for word in self.joy_keywords if word in text)
        }
        
        if any(emotion_counts.values()):
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
            if dominant_emotion[1] > 0:
                return dominant_emotion[0]
        
        return "none"