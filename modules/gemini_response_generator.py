import logging
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import asyncio
from config import Config
from modules.gemini_prompt_engineering import MentalHealthPromptEngineering

class GeminiResponseGenerator:
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        self.mental_health_topics = Config.MENTAL_HEALTH_TOPICS
        genai.configure(api_key=self.api_key)
        
        # Set up the model
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # System prompt to constrain responses to mental health
        self.system_prompt = """
        You are a compassionate mental health support chatbot. Your purpose is to:
        1. Provide emotional support and information ONLY about mental health topics
        2. Respond with empathy to users experiencing emotional difficulties
        3. Offer healthy coping strategies when appropriate
        4. Direct to professional resources when needed
        
        Important constraints:
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
        self._last_source = "gemini"
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass
        
    async def generate_response(self, intent: str, sentiment: str, emotions: str, context: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> str:
        """Generate a response using Gemini API while ensuring it stays within mental health domain"""
        
        # Extract the last few messages for context
        conversation_history = self._format_conversation_history(context)
        
        # Get the last user message
        last_user_msg = next((msg for msg in reversed(context) if msg.get('role') == 'user'), None)
        user_input = last_user_msg.get('content', '') if last_user_msg else ""
        
        # Build the specialized mental health prompt
        specialized_prompt = MentalHealthPromptEngineering.create_empathetic_prompt(
            intent, emotions, context, user_profile
        )
        
        # Combine all prompts
        style = user_profile.get('preferred_responses', 'neutral')
        emotion_str = f"The user is feeling {emotions}." if emotions != "none" else ""
        intent_str = f"The user's intent is {intent}." if intent != "general" else "The user's intent is unclear."
        
        prompt = f"""
        {self.system_prompt}
        
        {specialized_prompt}
        
        Response style preference: {style}
        {emotion_str} 
        {intent_str}
        
        Conversation history:
        {conversation_history}
        
        Current user message: {user_input}
        
        Respond compassionately to this mental health concern.
        """
        
        try:
            # Generate response with Gemini
            result = await self._async_generate(prompt)
            self._last_source = "gemini"
            
            # Safety check for crisis situations
            if intent == "crisis" or self._contains_crisis_language(user_input):
                if "988" not in result and "crisis" not in result.lower():
                    result += "\n\nIf you're in crisis, please call 988 for immediate support."
                
            # Limit response length
            if len(result) > 500:
                result = result[:497] + "..."
                
            return result
            
        except Exception as e:
            logging.error(f"Gemini response generation error: {str(e)}")
            return "I'm sorry, I'm having trouble processing that right now. How are you feeling today?"
            
    async def _async_generate(self, prompt: str) -> str:
        """Wrapper for asynchronous generation with Gemini"""
        try:
            # Using a synchronous call in an async wrapper since Google's API might not have native async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.model.generate_content(prompt))
            return response.text
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return "I'm having trouble connecting. Can we try again?"
            
    def _format_conversation_history(self, context: List[Dict[str, Any]]) -> str:
        """Format the conversation history for the prompt"""
        formatted = ""
        # Take the last 5 messages to avoid token limits
        for msg in context[-5:]:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role and content:
                role_name = "User" if role == "user" else "Assistant"
                formatted += f"{role_name}: {content}\n"
        return formatted
        
    def _contains_crisis_language(self, text: str) -> bool:
        """Check if text contains crisis indicators"""
        crisis_keywords = Config.CRISIS_KEYWORDS
        text_lower = text.lower()
        
        return any(keyword in text_lower for keyword in crisis_keywords)