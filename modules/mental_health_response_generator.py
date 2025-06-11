import logging
import json
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import asyncio
import httpx
import re
from config import Config
from modules.gemini_prompt_engineering import MentalHealthPromptEngineering

class MentalHealthResponseGenerator:
    def __init__(self):
        # Chatbot name
        self.chatbot_name = "NeuralEase"
        
        # API keys
        self.gemini_api_key = Config.GEMINI_API_KEY
        self.openai_api_key = Config.OPENAI_API_KEY
        self.mental_health_topics = Config.MENTAL_HEALTH_TOPICS
        
        # FIXED: Removed persistent HTTP client that was causing "client closed" errors
        # self.http_client = httpx.AsyncClient(timeout=30.0)  # ← REMOVED THIS LINE
        
        # Set up Gemini with proper configuration
        if self.gemini_api_key:
            try:
                # Configure the Google generativeai library
                genai.configure(api_key=self.gemini_api_key)
                
                # Set fixed model based on your working test
                self.gemini_model_name = "gemini-2.0-flash"
                
                # Set up direct API endpoint - this matches your successful test format
                self.direct_api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_name}:generateContent?key={self.gemini_api_key}"
                logging.info(f"✅ Gemini API configured with key: {self.gemini_api_key[:10]}...")
                logging.info(f"✅ Direct API endpoint ready: {self.direct_api_endpoint}")
                
                # Try to initialize the model using the library as fallback
                try:
                    self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                    logging.info(f"✅ Library model initialized: {self.gemini_model_name}")
                except Exception as model_error:
                    logging.warning(f"⚠️ Could not initialize model through library: {str(model_error)}")
                    self.gemini_model = None
                    
            except Exception as e:
                logging.error(f"❌ Failed to initialize Gemini: {str(e)}")
                self.gemini_model = None
                self.gemini_model_name = None
        else:
            logging.warning("❌ No Gemini API key provided")
            self.gemini_model = None
            self.gemini_model_name = None
        
        # Set up OpenAI as secondary fallback
        self.openai_client = None
        if self.openai_api_key:
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
                logging.info(f"✅ OpenAI client initialized with key: {self.openai_api_key[:10]}...")
            except Exception as e:
                logging.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
        else:
            logging.warning("❌ No OpenAI API key provided")
        
        # Base system prompt with mental health focus and NeuralEase branding
        self.base_system_prompt = f"""
        You are {self.chatbot_name}, a compassionate mental health support chatbot. Your purpose is to:
        1. Provide emotional support and information ONLY about mental health topics
        2. Respond with empathy to users experiencing emotional difficulties
        3. Offer healthy coping strategies when appropriate
        4. Direct to professional resources when needed
        
        Important constraints:
        - Always refer to yourself as "{self.chatbot_name}" when introducing yourself
        - NEVER provide medical diagnoses or treatment advice
        - ALWAYS recommend professional help for crisis situations
        - ONLY respond to mental health related queries
        - For non-mental health topics, politely redirect the conversation to mental health
        - Keep responses concise (100 words maximum)
        - For crisis situations, include the 988 Suicide & Crisis Lifeline
        
        Safety protocols:
        - If the user indicates intent to harm themselves or others, emphasize seeking immediate help
        - Do not engage with explicit content, hate speech, or illegal activities
        """
        
        # Enhanced system prompt for OpenAI to ensure mental health focus
        self.openai_system_prompt = f"""
        {self.base_system_prompt}
        
        CRITICAL INSTRUCTIONS FOR OPENAI:
        - You are {self.chatbot_name}, a mental health support chatbot
        - You are STRICTLY LIMITED to mental health topics only
        - If asked about ANY non-mental health topic (e.g., sports, news, technology, cooking, etc.), 
          respond with: "As {self.chatbot_name}, I'm focused on providing mental health support. Is there something about 
          your mental or emotional wellbeing you'd like to discuss?"
        - Your ONLY purpose is mental health support - do not answer questions outside this domain
        - Focus on emotional wellbeing, coping strategies, stress management, and general mental health information
        - NEVER provide specific medical advice, diagnoses, or treatment recommendations
        - For crisis situations, ALWAYS include the 988 Crisis Lifeline information
        
        Remember: You are {self.chatbot_name} and you are ONLY permitted to discuss mental health related topics.
        """
        
        # Track response source
        self._last_source = "builtin"
        
        # Load built-in fallback responses
        self._fallback_responses = self._load_fallback_responses()
        
    def _load_fallback_responses(self):
        """Load built-in responses for when all APIs fail"""
        return {
            "greeting": [
                f"Hello! I'm {self.chatbot_name}, here to support you with mental health concerns. How are you feeling today?",
                f"Hi there. I'm {self.chatbot_name}, your mental health assistant. How can I help you today?",
                f"Welcome to {self.chatbot_name}. I'm here to listen and support you. How are you doing right now?"
            ],
            "emotional_support": [
                "I hear that you're going through a difficult time. It's okay to feel this way, and you're not alone. Would you like to talk more about what you're experiencing?",
                "That sounds really challenging. Many people experience similar feelings, and it's completely valid to feel this way. What helps you cope when you feel like this?",
                "I'm sorry you're feeling this way. Your emotions are valid, and it takes courage to express them. Would you like to explore some strategies that might help?"
            ],
            "coping_strategies": [
                "Some strategies that might help include deep breathing, mindfulness, gentle physical activity, or talking with a trusted person. Would you like to know more about any of these?",
                "When feeling overwhelmed, many find it helpful to practice grounding techniques, like the 5-4-3-2-1 method where you notice 5 things you see, 4 things you feel, and so on. Would you like to try this?",
                "Creating a self-care routine can be helpful. This might include regular sleep, balanced nutrition, movement, and time for activities you enjoy. What self-care activities resonate with you?"
            ],
            "crisis": [
                "I'm concerned about what you've shared. If you're in crisis, please call 988 for immediate support from the Suicide & Crisis Lifeline. They're available 24/7 and can help you through this difficult time.",
                "Your safety is important. Please reach out to the 988 Suicide & Crisis Lifeline right away by calling or texting 988. They provide free, confidential support 24/7.",
                "This sounds serious. Please contact crisis support immediately by calling 988. Professional help is available, and you deserve immediate support for what you're experiencing."
            ],
            "seeking_information": [
                "Mental health is about our emotional, psychological, and social well-being. It affects how we think, feel, and act. What specific aspect would you like to know more about?",
                "There are many resources available for mental health support. These include therapy, support groups, self-help strategies, and crisis services. Would you like information about any of these?",
                "Understanding mental health is an important step in maintaining wellbeing. Is there a particular topic or condition you'd like to learn more about?"
            ],
            "resources_request": [
                "For mental health resources, the National Institute of Mental Health (nimh.nih.gov) and SAMHSA (samhsa.gov) offer reliable information. For immediate support, the 988 Suicide & Crisis Lifeline is available 24/7.",
                "There are many resources available, including online therapy platforms, community mental health centers, and support groups. Which type of resource would be most helpful for you right now?",
                "Mental health resources include crisis lines like 988, therapy services, support groups, and educational websites. What kind of support are you looking for specifically?"
            ],
            "general": [
                f"I'm {self.chatbot_name}, here to support you with mental health concerns. What's on your mind today?",
                "Mental wellbeing is important. How can I help support yours today?",
                f"This is {self.chatbot_name}, focused on helping with emotional and mental health. What would be most helpful for you right now?"
            ]
        }
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # FIXED: Removed HTTP client closure since we don't have a persistent client anymore
        # await self.http_client.aclose()  # ← REMOVED THIS LINE
        pass
        
    async def generate_response(self, intent: str, sentiment: str, emotions: str, context: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> str:
        """Generate a response using cascading fallback system:
        1. Try Gemini API first (primary) - YOUR WORKING API
        2. Fall back to OpenAI if Gemini fails (secondary)
        3. Use built-in responses if both APIs fail (tertiary)
        """
        # Extract context information
        conversation_history = self._format_conversation_history(context)
        last_user_msg = next((msg for msg in reversed(context) if msg.get('role') == 'user'), None)
        user_input = last_user_msg.get('content', '') if last_user_msg else ""
        style = user_profile.get('preferred_responses', 'neutral')
        
        # Build the specialized prompt
        specialized_prompt = MentalHealthPromptEngineering.create_empathetic_prompt(
            intent, emotions, context, user_profile
        )
        
        # Combine prompts for Gemini
        emotion_str = f"The user is feeling {emotions}." if emotions != "none" else ""
        intent_str = f"The user's intent is {intent}." if intent != "general" else "The user's intent is unclear."
        
        gemini_prompt = f"""
        {self.base_system_prompt}
        
        {specialized_prompt}
        
        Response style preference: {style}
        {emotion_str} 
        {intent_str}
        
        Conversation history:
        {conversation_history}
        
        Current user message: {user_input}
        
        Respond as {self.chatbot_name}, providing compassionate mental health support.
        """
        
        # 1. PRIMARY: Try direct Gemini API call first (YOUR WORKING API!)
        if self.gemini_api_key:
            try:
                logging.info("🚀 PRIMARY: Attempting Gemini API call (your working API)")
                result = await self._async_generate_gemini_direct(gemini_prompt)
                self._last_source = "gemini"
                
                # Add crisis resources if needed
                if intent == "crisis" or self._contains_crisis_language(user_input):
                    if "988" not in result and "crisis" not in result.lower():
                        result += "\n\nIf you're in crisis, please call 988 for immediate support."
                
                # Verify the response is mental health focused
                if not self._is_mental_health_response(result):
                    logging.warning("Gemini response was not mental health focused - applying correction")
                    result = self._apply_mental_health_correction(result)
                
                # If first message, ensure chatbot introduces itself as NeuralEase
                if self._is_initial_greeting(context):
                    if self.chatbot_name not in result:
                        result = f"Hi, I'm {self.chatbot_name}! " + result
                
                # Limit response length
                if len(result) > 500:
                    result = result[:497] + "..."
                    
                logging.info(f"✅ SUCCESS: Gemini API response generated successfully")
                return result
                
            except Exception as e:
                logging.error(f"❌ Gemini API call failed: {str(e)}")
                logging.info("⏭️ Falling back to OpenAI (secondary)")
        else:
            logging.warning("❌ No Gemini API key, skipping to OpenAI (secondary)")
            
        # 2. SECONDARY: Try OpenAI as fallback
        if self.openai_client:
            try:
                logging.info("🔄 SECONDARY: Attempting OpenAI generation")
                result = await self._async_generate_openai(user_input, conversation_history, intent, emotions, style)
                self._last_source = "openai"
                
                # Add crisis resources if needed
                if intent == "crisis" or self._contains_crisis_language(user_input):
                    if "988" not in result and "crisis" not in result.lower():
                        result += "\n\nIf you're in crisis, please call 988 for immediate support."
                
                # Verify the response is mental health focused
                if not self._is_mental_health_response(result):
                    logging.warning("OpenAI response was not mental health focused - applying correction")
                    result = self._apply_mental_health_correction(result)
                
                # If first message, ensure chatbot introduces itself as NeuralEase
                if self._is_initial_greeting(context):
                    if self.chatbot_name not in result:
                        result = f"Hi, I'm {self.chatbot_name}! " + result
                    
                # Limit response length
                if len(result) > 500:
                    result = result[:497] + "..."
                    
                logging.info(f"✅ SUCCESS: OpenAI response generated successfully")
                return result
                
            except Exception as e:
                logging.error(f"❌ OpenAI generation failed: {str(e)}")
                logging.info("⏭️ Falling back to built-in responses (tertiary)")
        else:
            logging.warning("❌ No OpenAI client, skipping to built-in responses (tertiary)")

        # 3. TERTIARY: Fall back to built-in responses as last resort
        logging.info("🛡️ TERTIARY: Using built-in fallback response")
        return self._get_fallback_response(intent, emotions, user_profile, context)
    
    def _is_initial_greeting(self, context: List[Dict[str, Any]]) -> bool:
        """Check if this is likely the first greeting from the system"""
        system_messages = [msg for msg in context if msg.get('role') == 'system']
        return len(system_messages) <= 1
    
    def _is_mental_health_response(self, response: str) -> bool:
        """
        Verify that a response is focused on mental health topics.
        This is a simple check that could be enhanced with more sophisticated verification.
        """
        # Count mental health related terms in the response
        mental_health_term_count = sum(1 for term in self.mental_health_topics if term in response.lower())
        
        # Check for common non-mental health topics that might indicate off-topic responses
        non_mental_health_patterns = [
            r'\b(stock|investment|market|crypto|bitcoin|financial advice)\b',
            r'\b(sports|game|team|match|player|score)\b',
            r'\b(recipe|cook|bake|food|meal|ingredient)\b',
            r'\b(movie|film|show|actor|watch|series)\b',
            r'\b(politics|election|vote|democrat|republican|policy)\b'
        ]
        
        # If we find non-mental health topics, it might be off-topic
        for pattern in non_mental_health_patterns:
            if re.search(pattern, response.lower()):
                return False
                
        # If we have enough mental health terms or the response is short, it's likely on-topic
        return mental_health_term_count >= 2 or len(response.split()) < 30
    
    def _apply_mental_health_correction(self, response: str) -> str:
        """
        Apply a correction to responses that might have strayed from mental health topics.
        """
        mental_health_redirection = (
            f"As {self.chatbot_name}, I'm focused specifically on mental health support. "
            "I'd be happy to discuss your emotional wellbeing, stress management, "
            "or other mental health topics. How are you feeling today?"
        )
        
        # Check if the response is very short (likely already a redirection)
        if len(response.split()) < 20:
            return mental_health_redirection
            
        # Otherwise try to preserve any useful content while redirecting
        shortened_response = ' '.join(response.split()[:20]) + '...'
        return f"I need to focus on mental health topics. {mental_health_redirection}"
            
    def _get_fallback_response(self, intent: str, emotions: str, user_profile: Dict[str, Any], context: List[Dict[str, Any]]) -> str:
        """Get a suitable fallback response when all APIs fail"""
        import random
        
        # Map intent to response category, defaulting to general
        category = intent if intent in self._fallback_responses else "general"
        
        # For grief, use emotional_support
        if emotions == "grief":
            category = "emotional_support"
            
        # For crisis, always use crisis category
        if intent == "crisis" or "crisis" in user_profile.get("last_input", "").lower():
            category = "crisis"
        
        # For initial greeting, always use greeting response
        if self._is_initial_greeting(context):
            category = "greeting"
            
        # Select a random response from the appropriate category
        responses = self._fallback_responses.get(category, self._fallback_responses["general"])
        response = random.choice(responses)
        
        self._last_source = "builtin"
        return response
    
    async def _async_generate_gemini_direct(self, prompt: str) -> str:
        """Generate text with Gemini using direct API call (FIXED HTTP CLIENT!)"""
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not configured")
            
        try:
            # Create the request payload - exactly like your working test
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            # Log the endpoint for debugging
            logging.debug(f"Using Gemini API endpoint: {self.direct_api_endpoint}")
            
            # FIXED: Create a fresh HTTP client for each request (prevents "client closed" error)
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Make the API call - format matches your working test
                headers = {"Content-Type": "application/json"}
                response = await client.post(
                    self.direct_api_endpoint,
                    headers=headers,
                    json=payload
                )
                
                # Check for errors
                if response.status_code != 200:
                    logging.error(f"Gemini API error: {response.status_code} - {response.text}")
                    raise Exception(f"API error: {response.status_code} - {response.text}")
                    
                # Parse the response
                result = response.json()
                
                # Extract the text from the response - matches your test response structure
                if (
                    "candidates" in result 
                    and len(result["candidates"]) > 0 
                    and "content" in result["candidates"][0]
                    and "parts" in result["candidates"][0]["content"]
                    and len(result["candidates"][0]["content"]["parts"]) > 0
                    and "text" in result["candidates"][0]["content"]["parts"][0]
                ):
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    logging.info(f"✅ Gemini API returned: {text[:50]}...")
                    return text
                else:
                    logging.error(f"❌ Unexpected Gemini response format: {result}")
                    raise Exception(f"Unexpected response format: {result}")
                
        except Exception as e:
            logging.error(f"❌ Direct Gemini API error: {str(e)}")
            raise
    
    async def _async_generate_openai(self, user_input: str, conversation_history: str, intent: str, emotions: str, style: str) -> str:
        """
        Generate text with OpenAI as secondary fallback,
        with ENHANCED mental health focus and safeguards
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # Create a more structured prompt for OpenAI
            user_prompt = f"""
            Intent: {intent}
            Emotion: {emotions}
            Style preference: {style}
            
            Recent conversation:
            {conversation_history}
            
            Current message: {user_input}
            
            Remember to respond as {self.chatbot_name}, a mental health support chatbot.
            """
            
            # Use the enhanced system prompt specifically designed for OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.openai_system_prompt},  # Enhanced system prompt
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            
            # Extra safeguard: Check if the response mentions being constrained to mental health
            constraint_phrases = [
                "I can only discuss mental health",
                "I'm focused on mental health",
                "I'm designed to provide mental health",
                "I can't help with that",
                "limited to mental health"
            ]
            
            # If response explicitly mentions constraints, rewrite it to be more natural
            for phrase in constraint_phrases:
                if phrase.lower() in result.lower():
                    return (f"As {self.chatbot_name}, I'm here to support you with mental health concerns. "
                            "Would you like to discuss how you're feeling emotionally or "
                            "explore strategies for mental wellbeing?")
            
            return result
            
        except Exception as e:
            logging.error(f"❌ OpenAI API error: {str(e)}")
            raise
            
    def _format_conversation_history(self, context: List[Dict[str, Any]]) -> str:
        """Format the conversation history for the prompt"""
        formatted = ""
        # Take the last 5 messages to avoid token limits
        for msg in context[-5:]:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role and content:
                role_name = "User" if role == "user" else f"{self.chatbot_name}"
                formatted += f"{role_name}: {content}\n"
        return formatted
        
    def _contains_crisis_language(self, text: str) -> bool:
        """Check if text contains crisis indicators"""
        crisis_keywords = Config.CRISIS_KEYWORDS
        text_lower = text.lower()
        
        return any(keyword in text_lower for keyword in crisis_keywords)