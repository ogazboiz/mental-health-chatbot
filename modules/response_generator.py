import json
import re
import random
from typing import Dict, List, Optional, Any
import logging
import asyncio
import httpx
from Bio import Entrez
from config import Config
from openai import AsyncOpenAI

class ResponseGenerator:
    def __init__(self):
        self.resources = Config.RESOURCE_LINKS
        self.neuroscience_terms = [
            "amygdala", "hippocampus", "prefrontal cortex", "limbic system", "cerebral cortex",
            "dopamine", "serotonin", "norepinephrine", "gaba", "glutamate", "depression", "anxiety"
        ]
        self.health_keywords = [
            "sleep", "insomnia", "fatigue", "headache", "pain", "stress", "anxiety", "depression"
        ]
        self.grief_keywords = ["lost", "loss", "died", "death", "grief", "bereavement", "passed", "gone"]
        self.emotional_keywords = ["sad", "anxious", "depressed", "down", "upset", "lonely", "worthless", "stressed"]
        self.coping_keywords = ["cope", "coping", "ways", "strategies", "deal", "manage"]
        self.greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "good night"]
        self.http_client = httpx.AsyncClient()
        self.openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        
        try:
            with open('responses.json', 'r', encoding='utf-8') as f:
                raw_responses = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load responses.json: {str(e)}")
            raise

        intent_map = {
            "0": "greeting",
            "1": "seeking_information",
            "2": "emotional_support",
            "3": "coping_strategies",
            "4": "resources_request",
            "5": "personal_story",
            "6": "crisis",
            "7": "physical_symptom"
        }
        
        self.kaggle_responses = {
            intent: raw_responses.get(label, ["Default response: I'm here to help. Could you share more?"])
            for label, intent in intent_map.items()
        }
        
        for intent, responses in self.kaggle_responses.items():
            logging.info(f"Loaded {len(responses)} responses for intent: {intent}")

        Entrez.email = "your.email@example.com"
        self._last_source = "default"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.http_client.aclose()

    async def _generate_openai_response(self, intent: str, emotions: str, context: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> Optional[str]:
        try:
            context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context[-3:]])
            emotion_str = f"The user is feeling {emotions}." if emotions != "none" else ""
            intent_str = f"The user’s intent is {intent}." if intent != "general" else "The user’s intent is unclear."
            style = user_profile.get('preferred_responses', 'neutral')
            prompt = f"""
            You are an empathetic mental health chatbot. Provide a supportive, safe response in a {style} tone.
            {emotion_str} {intent_str}
            Conversation history:
            {context_str}
            Current input: {user_profile.get('last_input', '')}
            Respond in up to 100 words. If crisis intent, mention 988. Avoid medical advice.
            """
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            self._last_source = "openai"
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Open AI response generation error: {str(e)}")
            self._last_source = "default"
            return None

    def _is_greeting(self, content: str) -> bool:
        return any(kw in content.lower() for kw in self.greeting_keywords)

    def _is_grief_related(self, content: str) -> bool:
        return any(kw in content.lower() for kw in self.grief_keywords)

    def _is_coping_request(self, content: str) -> bool:
        return any(kw in content.lower() for kw in self.coping_keywords)

    def _extract_conversation_history(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        history = []
        for msg in context:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                logging.warning(f"Invalid context message: {msg}")
                continue
            history.append({
                'role': msg['role'],
                'content': msg['content'],
                'metadata': msg.get('metadata', {})
            })
        return history[-Config.CONTEXT_WINDOW:]

    async def _web_search(self, query: str) -> Optional[str]:
        try:
            response = await self.http_client.get(
                f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            return data.get("AbstractText", None)
        except Exception as e:
            logging.error(f"Web search error: {str(e)}")
            return None

    async def _query_pubmed_api(self, term: str) -> Optional[str]:
        try:
            handle = Entrez.esearch(db="pubmed", term=term, retmax=1)
            record = Entrez.read(handle)
            handle.close()
            if record["IdList"]:
                handle = Entrez.esummary(db="pubmed", id=record["IdList"][0])
                summary = Entrez.read(handle)
                handle.close()
                return summary[0].get("Title", None)
            return None
        except Exception as e:
            logging.error(f"PubMed API error: {str(e)}")
            return None

    async def _handle_greeting(self, context: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> str:
        style = user_profile.get('preferred_responses', 'neutral')
        greeting_responses = self.kaggle_responses.get("greeting", ["Hello! How can I assist you today?"])
        self._last_source = "kaggle"
        if style == "friendly":
            return f"Hey! Good to hear from you! 😊 {random.choice(greeting_responses)}"
        return random.choice(greeting_responses)

    async def _handle_information_request(self, context: List[Dict[str, Any]], sentiment: str, emotions: str, user_profile: Dict[str, Any]) -> str:
        style = user_profile.get('preferred_responses', 'neutral')
        last_user_msg = next((msg for msg in reversed(context) if msg.get('role') == 'user'), None)
        content = last_user_msg.get('content', '').lower() if last_user_msg else ""
        keywords = last_user_msg.get('metadata', {}).get('keywords', []) if last_user_msg else []
        self._last_source = "kaggle"

        for kw in keywords:
            if kw in self.health_keywords or kw in self.grief_keywords:
                web_result = await self._web_search(kw + " mental health")
                if web_result:
                    if style == "professional":
                        return (
                            f"Regarding {kw}, {web_result}. Would you like more specific information? "
                            f"See {self.resources['general']} for additional resources."
                        )
                    elif style == "friendly":
                        return (
                            f"Hey, about {kw}, {web_result} 😊 Want to learn more? "
                            f"Check out {self.resources['general']} for extra info!"
                        )
                    return (
                        f"{kw} info: {web_result}. Interested in more details? "
                        f"Explore {self.resources['general']}."
                    )

        responses = self.kaggle_responses.get("seeking_information", ["I can provide information on various mental health topics."])
        response = random.choice(responses)
        if len(response) > 150 and style != "professional":
            response = response[:150] + "... Want the full details?"
        if style == "professional":
            return (
                f"{response} Could you specify what you'd like to know? "
                f"Resources are available at {self.resources['general']}."
            )
        return (
            f"{response} 😊 What specifically do you want to know? "
            f"Check out {self.resources['general']} for more."
        )

    async def _handle_emotional_support(self, context: List[Dict[str, Any]], sentiment: str, emotions: str, user_profile: Dict[str, Any]) -> str:
        style = user_profile.get('preferred_responses', 'neutral')
        last_user_msg = next((msg for msg in reversed(context) if msg.get('role') == 'user'), None)
        content = last_user_msg.get('content', '').lower() if last_user_msg else ""
        intent_confidence = last_user_msg.get('metadata', {}).get('intent', {}).get('confidence', 0.5) if last_user_msg else 0.5

        if intent_confidence < 0.7:
            openai_response = await self._generate_openai_response("emotional_support", emotions, context, user_profile)
            if openai_response:
                return openai_response

        if self._is_coping_request(content):
            return await self._handle_coping_strategies(context, sentiment, emotions, user_profile)

        if emotions == "grief" or self._is_grief_related(content):
            openai_response = await self._generate_openai_response("emotional_support", emotions, context, user_profile)
            if openai_response:
                return openai_response
            if style == "friendly":
                return (
                    f"I’m so sorry to hear about your loss—that’s really tough. 💙 "
                    "Thanks for trusting me with that. Want to share more or talk about ways to cope?"
                )
            elif style == "professional":
                return (
                    "I’m deeply sorry for your loss. Losing someone close is incredibly challenging. "
                    "Would you like to share more about your experience or explore grief support resources?"
                )
            return (
                "I’m very sorry about your loss. This must be a difficult time. "
                "Would you like to discuss your feelings or learn about ways to cope with grief?"
            )

        if emotions == "sadness" or any(kw in content for kw in self.emotional_keywords):
            openai_response = await self._generate_openai_response("emotional_support", emotions, context, user_profile)
            if openai_response:
                return openai_response
            responses = self.kaggle_responses.get("emotional_support", ["I’m here to listen."])
            if style == "friendly":
                return (
                    f"{random.choice(responses)} 😔 It’s okay to have these moments. "
                    "Want to talk more about what’s going on or try some calming ideas together?"
                )
            return (
                f"{random.choice(responses)} It can be tough sometimes. "
                "Would you like to share more or explore ways to feel better?"
            )

        openai_response = await self._generate_openai_response("emotional_support", emotions, context, user_profile)
        if openai_response:
            return openai_response
        return (
            f"I’m here to listen. Could you share a bit more about how you’re feeling? "
            f"You can also check out {self.resources['general']} for support."
        )

    async def _handle_coping_strategies(self, context: List[Dict[str, Any]], sentiment: str, emotions: str, user_profile: Dict[str, Any]) -> str:
        style = user_profile.get('preferred_responses', 'neutral')
        last_user_msg = next((msg for msg in reversed(context) if msg.get('role') == 'user'), None)
        content = last_user_msg.get('content', '').lower() if last_user_msg else ""
        coping_responses = self.kaggle_responses.get("coping_strategies", ["One way to manage stress is deep breathing."])
        self._last_source = "kaggle"

        if "sleep" in content or "insomnia" in content:
            if style == "friendly":
                return (
                    f"Trouble sleeping can be rough! 😴 Try a relaxing bedtime routine or avoiding screens before bed. "
                    f"Want more tips? Check out {self.resources['sleep']}!"
                )
            return (
                f"Sleep issues can be challenging. Establishing a consistent bedtime routine may help. "
                f"Would you like more strategies? See {self.resources['sleep']} for resources."
            )

        if "grief" in content or emotions == "grief":
            grief_coping_strategies = [
                "Consider journaling your thoughts and memories to process your grief. Would you like tips on how to start?",
                "Talking to a trusted friend or joining a support group can help. Would you like resources for finding support groups?",
                "Practicing self-care, like gentle exercise or meditation, can ease the pain. Want to try a simple mindfulness exercise?"
            ]
            if style == "friendly":
                return (
                    f"Coping with grief is so hard, and I’m here to help. 💙 {random.choice(grief_coping_strategies)} "
                    f"Check out {self.resources['general']} for more support."
                )
            return (
                f"Grief can be overwhelming, but there are ways to cope. {random.choice(grief_coping_strategies)} "
                f"See {self.resources['general']} for additional resources."
            )

        if style == "friendly":
            return (
                f"Let’s try something to help you feel better! 😊 {random.choice(coping_responses)} "
                f"Or check out {self.resources['general']} for more ideas!"
            )
        return (
            f"{random.choice(coping_responses)} Would you like more strategies? "
            f"Additional resources are available at {self.resources['general']}."
        )

    async def _handle_resources_request(self, context: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> str:
        style = user_profile.get('preferred_responses', 'neutral')
        last_user_msg = next((msg for msg in reversed(context) if msg.get('role') == 'user'), None)
        content = last_user_msg.get('content', '').lower() if last_user_msg else ""
        self._last_source = "kaggle"

        if "crisis" in content or "urgent" in content:
            if style == "friendly":
                return (
                    f"If you’re in a tough spot, you’re not alone. 💙 Reach out at {self.resources['crisis']} for immediate help. "
                    "Want to talk more?"
                )
            return (
                f"For immediate support, please contact {self.resources['crisis']}. "
                "Would you like to discuss further or explore other resources?"
            )

        responses = self.kaggle_responses.get("resources_request", ["You can find mental health resources at..."])
        if style == "friendly":
            return (
                f"{random.choice(responses)} 😊 Need something specific? Let me know!"
            )
        return (
            f"{random.choice(responses)} Would you like assistance finding specific support?"
        )

    async def _handle_personal_story(self, context: List[Dict[str, Any]], sentiment: str, emotions: str, user_profile: Dict[str, Any]) -> str:
        style = user_profile.get('preferred_responses', 'neutral')
        last_user_msg = next((msg for msg in reversed(context) if msg.get('role') == 'user'), None)
        content = last_user_msg.get('content', '').lower() if last_user_msg else ""
        intent_confidence = last_user_msg.get('metadata', {}).get('intent', {}).get('confidence', 0.5) if last_user_msg else 0.5

        if intent_confidence < 0.7:
            openai_response = await self._generate_openai_response("personal_story", emotions, context, user_profile)
            if openai_response:
                return openai_response

        if emotions == "grief" or self._is_grief_related(content):
            openai_response = await self._generate_openai_response("personal_story", emotions, context, user_profile)
            if openai_response:
                return openai_response
            if style == "professional":
                return (
                    f"Thank you for sharing about your loss. It’s understandable that this is a challenging time. "
                    "Would you like to talk more about your experience or explore grief support resources?"
                )
            elif style == "friendly":
                return (
                    f"I’m so sorry to hear about your loss—that’s really tough. 💙 Thanks for trusting me with that. "
                    "Want to share more or talk about ways to cope?"
                )
            return (
                f"I’m sorry for your loss. It’s okay to feel this way. "
                "Would you like to share more or learn about coping with grief?"
            )

        openai_response = await self._generate_openai_response("personal_story", emotions, context, user_profile)
        if openai_response:
            return openai_response
        responses = self.kaggle_responses.get("personal_story", ["Thank you for sharing your experience."])
        if style == "professional":
            return (
                f"{random.choice(responses)} It sounds important to you. "
                "Would you like to explore how you’re feeling further?"
            )
        return (
            f"{random.choice(responses)} 😊 Want to tell me more about it?"
        )

    async def _handle_crisis(self, context: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> str:
        style = user_profile.get('preferred_responses', 'neutral')
        self._last_source = "kaggle"
        responses = self.kaggle_responses.get("crisis", ["I’m concerned about your safety."])
        if style == "friendly":
            return (
                f"I’m really worried about you—let’s get you some help right away. 💙 "
                f"{random.choice(responses)} Want to talk more while you get support?"
            )
        return (
            f"{random.choice(responses)} Please contact {self.resources['crisis']} or call 988 immediately. "
            "Would you like to discuss further while seeking help?"
        )

    async def _handle_physical_symptom(self, context: List[Dict[str, Any]], sentiment: str, emotions: str, user_profile: Dict[str, Any]) -> str:
        style = user_profile.get('preferred_responses', 'neutral')
        last_user_msg = next((msg for msg in reversed(context) if msg.get('role') == 'user'), None)
        content = last_user_msg.get('content', '').lower() if last_user_msg else ""
        self._last_source = "kaggle"

        responses = self.kaggle_responses.get("physical_symptom", ["I’m not a medical professional, but I can offer general support."])
        if style == "friendly":
            return (
                f"{random.choice(responses)} 😊 Can you tell me more about what you’re feeling? "
                f"Check out {self.resources['general']} for general health info."
            )
        return (
            f"{random.choice(responses)} Please consult a doctor for physical symptoms. "
            f"Would you like to share more or explore resources at {self.resources['general']}?"
        )

    async def _handle_follow_up(self, history: List[Dict[str, Any]], intent: str, sentiment: str, emotions: str, user_profile: Dict[str, Any]) -> Optional[str]:
        style = user_profile.get('preferred_responses', 'neutral')
        last_user_msg = next((msg for msg in reversed(history) if msg.get('role') == 'user'), None)
        previous_system_msg = next((msg for msg in reversed(history[:-1]) if msg.get('role') == 'system'), None)
        
        if not last_user_msg:
            logging.debug("No valid user message for follow-up handling")
            return None
        
        content = last_user_msg.get('content', '').lower()
        prev_system_content = previous_system_msg.get('content', '').lower() if previous_system_msg else ""
        
        if self._is_coping_request(content):
            return await self._handle_coping_strategies(history, sentiment, emotions, user_profile)

        if emotions == "grief" or self._is_grief_related(content):
            openai_response = await self._generate_openai_response("emotional_support", emotions, history, user_profile)
            if openai_response:
                return openai_response
            if style == "friendly":
                return (
                    f"I’m so sorry to hear about your loss—that’s really tough. 💙 "
                    "Thanks for sharing. Would you like to talk more about what happened or how you’re feeling?"
                )
            elif style == "professional":
                return (
                    "I’m deeply sorry for your loss. Losing someone close is incredibly challenging. "
                    "Would you like to share more about your experience or explore grief support resources?"
                )
            return (
                "I’m very sorry about your loss. This must be a difficult time. "
                "Would you like to discuss your feelings or learn about ways to cope with grief?"
            )
        
        if "tell me more" in prev_system_content or "how do you feel" in prev_system_content:
            if any(word in content for word in self.emotional_keywords) or emotions in ["sadness", "grief"]:
                openai_response = await self._generate_openai_response("emotional_support", emotions, history, user_profile)
                if openai_response:
                    return openai_response
                if style == "friendly":
                    return (
                        f"I hear how tough this is for you. 😔 Thanks for opening up. "
                        "Would you like to share more or talk about some ways to feel a bit better?"
                    )
                return (
                    "Thank you for sharing how you’re feeling. It sounds really challenging. "
                    "Would you like to discuss this further or explore coping strategies?"
                )
        
        return None

    async def generate_response(self, intent: str, sentiment: str, emotions: str, context: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> str:
        history = self._extract_conversation_history(context)
        style = user_profile.get('preferred_responses', 'neutral')
        last_user_msg = next((msg for msg in reversed(history) if msg.get('role') == 'user'), None)
        content = last_user_msg.get('content', '').lower() if last_user_msg else ""
        intent_confidence = last_user_msg.get('metadata', {}).get('intent', {}).get('confidence', 0.5) if last_user_msg else 0.5

        if intent == "general" and self._is_greeting(content):
            logging.info(f"Detected greeting keyword in '{content}'")
            return await self._handle_greeting(history, user_profile)

        if intent == "general" or intent_confidence < 0.7:
            openai_response = await self._generate_openai_response("general", emotions, history, user_profile)
            if openai_response:
                return openai_response
            logging.info("Using kaggle responses for general intent")
            responses = self.kaggle_responses.get("general", ["I’m not sure I understood, but I’d love to help."])
            self._last_source = "kaggle"
            if style == "friendly":
                return (
                    f"{random.choice(responses)} 😊 Could you share a bit more? "
                    f"Or check out {self.resources['general']} for support!"
                )
            return (
                f"{random.choice(responses)} Could you clarify? "
                f"You can also explore {self.resources['general']} for resources."
            )

        if intent == "coping_strategies" or self._is_coping_request(content):
            return await self._handle_coping_strategies(history, sentiment, emotions, user_profile)

        if emotions == "grief" or self._is_grief_related(content):
            return await self._handle_emotional_support(history, sentiment, emotions, user_profile)

        if intent == "emotional_support" or emotions == "sadness":
            follow_up_response = await self._handle_follow_up(history, intent, sentiment, emotions, user_profile)
            if follow_up_response:
                return follow_up_response
            return await self._handle_emotional_support(history, sentiment, emotions, user_profile)

        if intent == "greeting":
            return await self._handle_greeting(history, user_profile)
        elif intent == "seeking_information":
            return await self._handle_information_request(history, sentiment, emotions, user_profile)
        elif intent == "resources_request":
            return await self._handle_resources_request(history, user_profile)
        elif intent == "personal_story":
            return await self._handle_personal_story(history, sentiment, emotions, user_profile)
        elif intent == "crisis":
            return await self._handle_crisis(history, user_profile)
        elif intent == "physical_symptom":
            return await self._handle_physical_symptom(history, sentiment, emotions, user_profile)

        openai_response = await self._generate_openai_response("general", emotions, history, user_profile)
        if openai_response:
            return openai_response
        self._last_source = "kaggle"
        responses = self.kaggle_responses.get("general", ["I’m not sure I understood, but I’d love to help."])
        if style == "friendly":
            return (
                f"{random.choice(responses)} 😊 Could you share a bit more? "
                f"Or check out {self.resources['general']} for support!"
            )
        return (
            f"{random.choice(responses)} Could you clarify? "
            f"You can also explore {self.resources['general']} for resources."
        )