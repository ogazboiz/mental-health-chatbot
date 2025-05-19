import asyncio
import os
import time
from flask import Flask, request, jsonify, make_response
from modules.conversation import Conversation
from modules.nlp_processor import NLPProcessor
from modules.mental_health_response_generator import MentalHealthResponseGenerator
from modules.safety_checker import SafetyChecker
from modules.mental_health_filter import MentalHealthFilter
from modules.user_auth import User, AuthToken
from config import Config
import logging
from datetime import datetime
import torch
from asgiref.wsgi import WsgiToAsgi
from asgiref.sync import async_to_sync
from functools import wraps

# Enforce WindowsSelectorEventLoopPolicy before any asyncio operations
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Setup logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger('').handlers = []
logging.getLogger('').addHandler(logging.FileHandler('chatbot.log'))
# Also log to console for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger('').addHandler(console_handler)

app = Flask(__name__)
nlp_processor = NLPProcessor()
response_generator = MentalHealthResponseGenerator()
safety_checker = SafetyChecker()
mental_health_filter = MentalHealthFilter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device set to use {device}")
logging.info("Mental Health Chatbot starting up with enhanced user features")

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in headers
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        # If no token in headers, check cookies
        if not token:
            token = request.cookies.get('token')
            
        # If no token in cookies, check JSON body
        if not token and request.is_json:
            token = request.json.get('token')
            
        if not token:
            return jsonify({'error': 'Authentication token is missing'}), 401
            
        # Validate token
        user_id = AuthToken.validate_token(token)
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 401
            
        # Load user
        user = User()
        if not user.load_by_user_id(user_id):
            return jsonify({'error': 'User not found'}), 404
            
        # Add user to kwargs
        kwargs['user'] = user
        
        return f(*args, **kwargs)
    
    return decorated

# User Authentication Endpoints
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({"error": "Username, email, and password are required"}), 400
            
        # Validate password strength
        if len(password) < 8:
            return jsonify({"error": "Password must be at least 8 characters long"}), 400
            
        # Create new user
        user = User()
        if not user.create_user(username, email, password):
            return jsonify({"error": "Username already exists"}), 409
            
        # Generate authentication token
        token = AuthToken.generate_token(user.user_id)
        
        # Return token with user information
        response = make_response(jsonify({
            "message": "User registered successfully",
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "token": token
        }))
        
        # Set token cookie
        response.set_cookie(
            'token', 
            token, 
            httponly=True, 
            max_age=Config.TOKEN_EXPIRY_HOURS * 3600,
            secure=Config.SECURE_COOKIES
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login a user"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400
            
        # Authenticate user
        user = User()
        if not user.authenticate(username, password):
            return jsonify({"error": "Invalid username or password"}), 401
            
        # Generate authentication token
        token = AuthToken.generate_token(user.user_id)
        
        # Return token with user information
        response = make_response(jsonify({
            "message": "Login successful",
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "token": token
        }))
        
        # Set token cookie
        response.set_cookie(
            'token', 
            token, 
            httponly=True, 
            max_age=Config.TOKEN_EXPIRY_HOURS * 3600,
            secure=Config.SECURE_COOKIES
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout a user by clearing the token cookie"""
    response = make_response(jsonify({"message": "Logout successful"}))
    response.delete_cookie('token')
    return response

@app.route('/api/auth/refresh', methods=['POST'])
@token_required
def refresh_token(user):
    """Refresh authentication token"""
    # Generate a new token
    token = AuthToken.generate_token(user.user_id)
    
    # Return new token
    response = make_response(jsonify({
        "message": "Token refreshed",
        "token": token
    }))
    
    # Set token cookie
    response.set_cookie(
        'token', 
        token, 
        httponly=True, 
        max_age=Config.TOKEN_EXPIRY_HOURS * 3600,
        secure=Config.SECURE_COOKIES
    )
    
    return response

# User Profile Endpoints
@app.route('/api/user/profile', methods=['GET'])
@token_required
def get_profile(user):
    """Get user profile"""
    return jsonify({
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "profile": user.profile,
        "created_at": user.created_at,
        "last_login": user.last_login
    })

@app.route('/api/user/profile', methods=['PUT'])
@token_required
def update_profile(user):
    """Update user profile"""
    try:
        data = request.get_json()
        profile_data = data.get('profile', {})
        
        # Update profile
        if user.update_profile(profile_data):
            return jsonify({
                "message": "Profile updated successfully",
                "profile": user.profile
            })
        else:
            return jsonify({"error": "Failed to update profile"}), 500
            
    except Exception as e:
        logging.error(f"Profile update error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Chat Session Management Endpoints
@app.route('/api/sessions', methods=['GET'])
@token_required
def get_sessions(user):
    """Get all chat sessions for a user"""
    sessions = user.get_all_sessions()
    return jsonify({
        "sessions": sessions,
        "count": len(sessions)
    })

@app.route('/api/sessions', methods=['POST'])
@token_required
def create_session(user):
    """Create a new chat session"""
    try:
        data = request.get_json()
        title = data.get('title', 'New Conversation')
        
        # Create a new conversation
        conversation = Conversation(Config.ENCRYPTION_KEY)
        conversation.set_user_id(user.user_id)
        conversation.set_title(title)
        conversation.set_consent(True)  # Auto-consent for logged in users
        
        # Add welcome message
        welcome_msg = f"Hello! I'm NeuralEase, a mental health support chatbot. How can I help you today?"
        conversation.add_message("system", welcome_msg, {"source": "welcome", "model": "builtin"})
        
        # Save the conversation
        conversation.save_session()
        
        # Add session to user's sessions
        user.add_session(conversation.session_id)
        
        return jsonify({
            "message": "Chat session created",
            "session_id": conversation.session_id,
            "title": conversation.title
        })
        
    except Exception as e:
        logging.error(f"Session creation error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/sessions/<session_id>', methods=['GET'])
@token_required
def get_session(user, session_id):
    """Get a specific chat session"""
    try:
        # Check if session belongs to user
        if session_id not in user.sessions:
            return jsonify({"error": "Session not found"}), 404
            
        # Load the session
        conversation = Conversation(Config.ENCRYPTION_KEY)
        if not conversation.load_session(session_id):
            return jsonify({"error": "Failed to load session"}), 500
            
        # Return session data
        return jsonify({
            "session_id": conversation.session_id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "last_interaction": conversation.last_interaction,
            "messages": conversation.messages,
            "user_profile": conversation.user_profile
        })
        
    except Exception as e:
        logging.error(f"Get session error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/sessions/<session_id>', methods=['PUT'])
@token_required
def update_session(user, session_id):
    """Update a chat session (rename)"""
    try:
        # Check if session belongs to user
        if session_id not in user.sessions:
            return jsonify({"error": "Session not found"}), 404
            
        data = request.get_json()
        title = data.get('title')
        
        if not title:
            return jsonify({"error": "Title is required"}), 400
            
        # Load the session
        conversation = Conversation(Config.ENCRYPTION_KEY)
        if not conversation.load_session(session_id):
            return jsonify({"error": "Failed to load session"}), 500
            
        # Update title
        conversation.set_title(title)
        conversation.save_session()
        
        return jsonify({
            "message": "Session updated",
            "session_id": session_id,
            "title": title
        })
        
    except Exception as e:
        logging.error(f"Update session error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@token_required
def delete_session(user, session_id):
    """Delete a chat session"""
    try:
        # Check if session belongs to user
        if session_id not in user.sessions:
            return jsonify({"error": "Session not found"}), 404
            
        # Load the session
        conversation = Conversation(Config.ENCRYPTION_KEY)
        if not conversation.load_session(session_id):
            return jsonify({"error": "Failed to load session"}), 500
            
        # Mark as deleted
        conversation.mark_deleted()
        
        # Remove from user's sessions
        user.remove_session(session_id)
        
        return jsonify({
            "message": "Session deleted",
            "session_id": session_id
        })
        
    except Exception as e:
        logging.error(f"Delete session error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Message Management Endpoints
@app.route('/api/sessions/<session_id>/messages', methods=['POST'])
@token_required
def send_message(user, session_id):
    """Send a message to a chat session"""
    # Use async_to_sync to run the async function properly
    return async_to_sync(_async_send_message)(user, session_id)

# Async implementation moved to a separate function
async def _async_send_message(user, session_id):
    """Async implementation of send_message"""
    start_time = datetime.now()
    try:
        # Check if session belongs to user
        if session_id not in user.sessions:
            return jsonify({"error": "Session not found"}), 404
            
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
            
        # Load the session
        conversation = Conversation(Config.ENCRYPTION_KEY)
        if not conversation.load_session(session_id):
            return jsonify({"error": "Failed to load session"}), 500
            
        logging.debug(f"Session {session_id}: Received input: {user_input}")

        # Safety check
        if not safety_checker.is_safe(user_input):
            logging.warning(f"Session {session_id}: Unsafe input detected: {user_input}")
            response = "I'm sorry, but that input contains inappropriate content. Please rephrase."
            conversation.add_message("user", user_input, {"is_safe": False})
            conversation.add_message("system", response, None)
            conversation.save_session()
            return jsonify({
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_profile": conversation.get_user_profile()
            })

        # Mental health domain check
        if not mental_health_filter.is_mental_health_related(user_input):
            logging.info(f"Session {session_id}: Non-mental health query detected: {user_input}")
            response = mental_health_filter.get_redirection_message(user_input)
            conversation.add_message("user", user_input, {"is_mental_health": False})
            conversation.add_message("system", response, {"source": "filter", "model": "rule-based"})
            conversation.save_session()
            return jsonify({
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_profile": conversation.get_user_profile(),
                "filtered": True
            })

        # Analyze text
        analysis = await nlp_processor.analyze_text(user_input)
        logging.debug(f"Session {session_id}: Full analysis structure: {analysis}")

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

        if not isinstance(analysis, dict):
            logging.error(f"Session {session_id}: Invalid analysis structure: {analysis}")
            return jsonify({"error": "Invalid analysis result"}), 500

        intent_dict = analysis.get('intent', {})
        if not isinstance(intent_dict, dict):
            logging.error(f"Session {session_id}: Invalid intent structure: {intent_dict}")
            intent = "general"
            analysis['intent'] = {"label": "LABEL_5", "confidence": 0.5, "intent": "general"}
        else:
            intent = intent_dict.get('intent', intent_map.get(intent_dict.get('label', ''), "general"))

        sentiment_dict = analysis.get('sentiment', {})
        if not isinstance(sentiment_dict, dict):
            logging.error(f"Session {session_id}: Invalid sentiment structure: {sentiment_dict}")
            sentiment = "neutral"
            analysis['sentiment'] = {"label": "neutral", "confidence": 0.5}
        else:
            sentiment = sentiment_dict.get('label', 'neutral')

        emotions = analysis.get('emotions', 'none')
        if not isinstance(emotions, str):
            logging.error(f"Session {session_id}: Invalid emotions structure: {emotions}")
            emotions = 'none'
            analysis['emotions'] = 'none'

        context = conversation.get_context()
        if not isinstance(context, list):
            logging.error(f"Session {session_id}: Context is not a list: {context}")
            context = []
        else:
            valid_context = [msg for msg in context if isinstance(msg, dict) and 'role' in msg and 'content' in msg]
            context = valid_context
            logging.debug(f"Session {session_id}: Validated context: {context}")

        grief_keywords = ["lost", "loss", "died", "death", "grief", "bereavement", "passed", "gone"]
        emotional_keywords = ["sad", "anxious", "depressed", "down", "upset"]
        analysis['is_response_to'] = len(context) + 1
        if context:
            last_message = next((msg for msg in reversed(context) if msg['role'] == 'system'), None)
            if last_message and any(phrase in last_message['content'].lower() for phrase in ["tell me more", "how do you feel", "share more"]):
                if intent not in ["emotional_support", "personal_story"] or any(kw in user_input.lower() for kw in grief_keywords + emotional_keywords):
                    logging.info(f"Session {session_id}: Overriding intent to emotional_support for follow-up or emotional input")
                    intent = "emotional_support"
                    analysis['intent'] = {"label": "LABEL_2", "confidence": 0.9, "intent": "emotional_support"}
                if any(kw in user_input.lower() for kw in grief_keywords):
                    emotions = "grief"
                    sentiment = "negative"
                    analysis['emotions'] = "grief"
                    analysis['sentiment'] = {"label": "negative", "confidence": 0.9}
            elif any(kw in user_input.lower() for kw in grief_keywords + emotional_keywords):
                logging.info(f"Session {session_id}: Setting emotional_support intent for emotional input")
                intent = "emotional_support"
                analysis['intent'] = {"label": "LABEL_2", "confidence": 0.9, "intent": "emotional_support"}
                if any(kw in user_input.lower() for kw in grief_keywords):
                    emotions = "grief"
                    sentiment = "negative"
                    analysis['emotions'] = "grief"
                    analysis['sentiment'] = {"label": "negative", "confidence": 0.9}

        if intent_dict.get('confidence', 0.0) < 0.7:
            intent = "general"
            analysis['intent']['intent'] = "general"

        # Check for crisis terms and override intent if needed
        if mental_health_filter.contains_crisis_language(user_input):
            intent = "crisis"
            analysis['intent'] = {"label": "LABEL_6", "confidence": 0.9, "intent": "crisis"}
            logging.warning(f"Session {session_id}: Crisis language detected, overriding intent to crisis")

        conversation.add_message("user", user_input, analysis)
        context = conversation.get_context()
        context = [msg for msg in context if isinstance(msg, dict) and 'role' in msg and 'content' in msg]

        user_profile = conversation.get_user_profile()
        if not isinstance(user_profile, dict):
            user_profile = {"preferred_responses": "neutral", "name": "", "emotion_history": [], "sentiment_history": []}
        user_profile["last_input"] = user_input

        # Generate response using cascading fallback system
        async with response_generator:
            response = await response_generator.generate_response(intent, sentiment, emotions, context, user_profile)

        # Add metadata about model used
        system_metadata = {
            "intent": analysis['intent'],
            "source": response_generator._last_source,
            "model": response_generator._last_source  # gemini_direct, gemini_library, openai, or builtin
        }
        conversation.add_message("system", response, system_metadata)
        conversation.save_session()

        response_time = (datetime.now() - start_time).total_seconds()
        logging.info(
            f"Session {session_id}: Input='{user_input}', Intent={intent}, Terms={analysis.get('neuroscience_terms', [])}"
            f", Emotions={emotions}, Sentiment={sentiment}"
        )
        logging.info(
            f"Session {session_id}: Response='{response[:50]}...', ResponseTime={response_time}s, ResponseSource={response_generator._last_source}"
        )

        return jsonify({
            "response": response,
            "message_id": conversation.messages[-1]["id"],  # Return the ID of the system message
            "session_id": session_id,
            "analysis": analysis,
            "user_profile": user_profile,
            "source": response_generator._last_source,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Send message error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/sessions/<session_id>/messages/<message_id>', methods=['PUT'])
@token_required
def edit_message(user, session_id, message_id):
    """Edit a message in a chat session"""
    try:
        # Check if session belongs to user
        if session_id not in user.sessions:
            return jsonify({"error": "Session not found"}), 404
            
        data = request.get_json()
        new_content = data.get('content')
        
        if not new_content:
            return jsonify({"error": "No content provided"}), 400
            
        # Load the session
        conversation = Conversation(Config.ENCRYPTION_KEY)
        if not conversation.load_session(session_id):
            return jsonify({"error": "Failed to load session"}), 500
            
        # Edit the message
        if not conversation.edit_message(message_id, new_content):
            return jsonify({"error": "Message not found"}), 404
            
        # Save the session
        conversation.save_session()
        
        return jsonify({
            "message": "Message edited",
            "session_id": session_id,
            "message_id": message_id
        })
        
    except Exception as e:
        logging.error(f"Edit message error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/sessions/<session_id>/messages/<message_id>', methods=['DELETE'])
@token_required
def delete_message(user, session_id, message_id):
    """Delete a message in a chat session"""
    try:
        # Check if session belongs to user
        if session_id not in user.sessions:
            return jsonify({"error": "Session not found"}), 404
            
        # Load the session
        conversation = Conversation(Config.ENCRYPTION_KEY)
        if not conversation.load_session(session_id):
            return jsonify({"error": "Failed to load session"}), 500
            
        # Delete the message
        if not conversation.delete_message(message_id):
            return jsonify({"error": "Message not found"}), 404
            
        # Save the session
        conversation.save_session()
        
        return jsonify({
            "message": "Message deleted",
            "session_id": session_id,
            "message_id": message_id
        })
        
    except Exception as e:
        logging.error(f"Delete message error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Consent, Feedback, and Status Endpoints (Keep these for backward compatibility)
@app.route('/consent', methods=['POST'])
def set_consent():
    try:
        data = request.get_json()
        session_id = data.get('session_id', '')
        consent = data.get('consent', False)
        
        conversation = Conversation(Config.ENCRYPTION_KEY)
        if session_id:
            if not conversation.load_session(session_id):
                logging.warning(f"Invalid session ID: {session_id}")
                return jsonify({"error": "Invalid session ID", "session_id": session_id}), 400
        else:
            session_id = conversation.session_id
        
        conversation.set_consent(consent)
        conversation.save_session()
        
        logging.info(f"Audit: Consent set to {consent} for session {session_id}")
        return jsonify({
            "message": "Consent updated",
            "session_id": session_id,
            "consent": consent
        })
    except Exception as e:
        logging.error(f"Consent endpoint error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        session_id = data.get('session_id', '')
        satisfaction = data.get('satisfaction', None)
        comments = data.get('comments', '')

        if not session_id or satisfaction is None:
            logging.warning(f"Invalid feedback request: session_id={session_id}, satisfaction={satisfaction}")
            return jsonify({"error": "Session ID and satisfaction score required"}), 400

        if not isinstance(satisfaction, int) or satisfaction < 1 or satisfaction > 5:
            logging.warning(f"Invalid satisfaction score: {satisfaction}")
            return jsonify({"error": "Satisfaction must be an integer between 1 and 5"}), 400

        conversation = Conversation(Config.ENCRYPTION_KEY)
        if not conversation.load_session(session_id):
            logging.warning(f"Invalid session ID for feedback: {session_id}")
            return jsonify({"error": "Invalid session ID", "session_id": session_id}), 400

        logging.info(f"Audit: Feedback received for session {session_id}: satisfaction={satisfaction}, comments={comments}")
        return jsonify({
            "message": "Feedback recorded",
            "session_id": session_id,
            "satisfaction": satisfaction,
            "comments": comments
        })
    except Exception as e:
        logging.error(f"Feedback endpoint error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Make the original /chat endpoint async-compatible too
@app.route('/chat', methods=['POST'])
def chat():
    """Legacy chat endpoint for backward compatibility"""
    return async_to_sync(_async_chat)()

async def _async_chat():
    """Async implementation of the legacy chat endpoint"""
    start_time = datetime.now()
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        session_id = data.get('session_id', '')

        if not user_input:
            logging.warning(f"Session {session_id}: Empty input received")
            return jsonify({"error": "No input provided", "session_id": session_id}), 400

        logging.debug(f"Session {session_id}: Received input: {user_input}")

        conversation = Conversation(Config.ENCRYPTION_KEY)
        if session_id:
            if not conversation.load_session(session_id):
                session_id = conversation.session_id
        else:
            session_id = conversation.session_id

        if not conversation.user_profile["consent_given"]:
            logging.warning(f"Session {session_id}: Consent required")
            return jsonify({
                "error": "Consent required",
                "session_id": session_id,
                "message": "Please provide consent to store conversation data via /consent endpoint"
            }), 403

        # Safety check
        if not safety_checker.is_safe(user_input):
            logging.warning(f"Session {session_id}: Unsafe input detected: {user_input}")
            response = "I'm sorry, but that input contains inappropriate content. Please rephrase."
            conversation.add_message("user", user_input, {"is_safe": False})
            conversation.add_message("system", response, None)
            conversation.save_session()
            return jsonify({
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_profile": conversation.get_user_profile()
            })

        # Mental health domain check
        if not mental_health_filter.is_mental_health_related(user_input):
            logging.info(f"Session {session_id}: Non-mental health query detected: {user_input}")
            response = mental_health_filter.get_redirection_message(user_input)
            conversation.add_message("user", user_input, {"is_mental_health": False})
            conversation.add_message("system", response, {"source": "filter", "model": "rule-based"})
            conversation.save_session()
            return jsonify({
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_profile": conversation.get_user_profile(),
                "filtered": True
            })

        analysis = await nlp_processor.analyze_text(user_input)
        logging.debug(f"Session {session_id}: Full analysis structure: {analysis}")

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

        if not isinstance(analysis, dict):
            logging.error(f"Session {session_id}: Invalid analysis structure: {analysis}")
            return jsonify({"error": "Invalid analysis result"}), 500

        intent_dict = analysis.get('intent', {})
        if not isinstance(intent_dict, dict):
            logging.error(f"Session {session_id}: Invalid intent structure: {intent_dict}")
            intent = "general"
            analysis['intent'] = {"label": "LABEL_5", "confidence": 0.5, "intent": "general"}
        else:
            intent = intent_dict.get('intent', intent_map.get(intent_dict.get('label', ''), "general"))

        sentiment_dict = analysis.get('sentiment', {})
        if not isinstance(sentiment_dict, dict):
            logging.error(f"Session {session_id}: Invalid sentiment structure: {sentiment_dict}")
            sentiment = "neutral"
            analysis['sentiment'] = {"label": "neutral", "confidence": 0.5}
        else:
            sentiment = sentiment_dict.get('label', 'neutral')

        emotions = analysis.get('emotions', 'none')
        if not isinstance(emotions, str):
            logging.error(f"Session {session_id}: Invalid emotions structure: {emotions}")
            emotions = 'none'
            analysis['emotions'] = 'none'

        context = conversation.get_context()
        if not isinstance(context, list):
            logging.error(f"Session {session_id}: Context is not a list: {context}")
            context = []
        else:
            valid_context = [msg for msg in context if isinstance(msg, dict) and 'role' in msg and 'content' in msg]
            context = valid_context
            logging.debug(f"Session {session_id}: Validated context: {context}")

        grief_keywords = ["lost", "loss", "died", "death", "grief", "bereavement", "passed", "gone"]
        emotional_keywords = ["sad", "anxious", "depressed", "down", "upset"]
        analysis['is_response_to'] = len(context) + 1
        if context:
            last_message = next((msg for msg in reversed(context) if msg['role'] == 'system'), None)
            if last_message and any(phrase in last_message['content'].lower() for phrase in ["tell me more", "how do you feel", "share more"]):
                if intent not in ["emotional_support", "personal_story"] or any(kw in user_input.lower() for kw in grief_keywords + emotional_keywords):
                    logging.info(f"Session {session_id}: Overriding intent to emotional_support for follow-up or emotional input")
                    intent = "emotional_support"
                    analysis['intent'] = {"label": "LABEL_2", "confidence": 0.9, "intent": "emotional_support"}
                if any(kw in user_input.lower() for kw in grief_keywords):
                    emotions = "grief"
                    sentiment = "negative"
                    analysis['emotions'] = "grief"
                    analysis['sentiment'] = {"label": "negative", "confidence": 0.9}
            elif any(kw in user_input.lower() for kw in grief_keywords + emotional_keywords):
                logging.info(f"Session {session_id}: Setting emotional_support intent for emotional input")
                intent = "emotional_support"
                analysis['intent'] = {"label": "LABEL_2", "confidence": 0.9, "intent": "emotional_support"}
                if any(kw in user_input.lower() for kw in grief_keywords):
                    emotions = "grief"
                    sentiment = "negative"
                    analysis['emotions'] = "grief"
                    analysis['sentiment'] = {"label": "negative", "confidence": 0.9}

        if intent_dict.get('confidence', 0.0) < 0.7:
            intent = "general"
            analysis['intent']['intent'] = "general"

        # Check for crisis terms and override intent if needed
        if mental_health_filter.contains_crisis_language(user_input):
            intent = "crisis"
            analysis['intent'] = {"label": "LABEL_6", "confidence": 0.9, "intent": "crisis"}
            logging.warning(f"Session {session_id}: Crisis language detected, overriding intent to crisis")

        conversation.add_message("user", user_input, analysis)
        context = conversation.get_context()
        context = [msg for msg in context if isinstance(msg, dict) and 'role' in msg and 'content' in msg]

        user_profile = conversation.get_user_profile()
        if not isinstance(user_profile, dict):
            user_profile = {"preferred_responses": "neutral", "name": "", "emotion_history": [], "sentiment_history": []}
        user_profile["last_input"] = user_input

        # Generate response using cascading fallback system
        async with response_generator:
            response = await response_generator.generate_response(intent, sentiment, emotions, context, user_profile)

        # Add metadata about model used
        system_metadata = {
            "intent": analysis['intent'],
            "source": response_generator._last_source,
            "model": response_generator._last_source  # gemini_direct, gemini_library, openai, or builtin
        }
        conversation.add_message("system", response, system_metadata)
        conversation.save_session()

        response_time = (datetime.now() - start_time).total_seconds()
        logging.info(
            f"Session {session_id}: Input='{user_input}', Intent={intent}, Terms={analysis.get('neuroscience_terms', [])}"
            f", Emotions={emotions}, Sentiment={sentiment}"
        )
        logging.info(
            f"Session {session_id}: Response='{response[:50]}...', ResponseTime={response_time}s, ResponseSource={response_generator._last_source}"
        )

        return jsonify({
            "response": response,
            "session_id": session_id,
            "analysis": analysis,
            "user_profile": user_profile,
            "source": response_generator._last_source,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to check API connectivity status"""
    status = {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "gemini_direct": "unknown",
            "gemini_library": "unknown",
            "openai": "unknown"
        }
    }
    
    # Check Gemini direct API status
    if response_generator.gemini_api_key:
        status["components"]["gemini_direct"] = "configured"
    else:
        status["components"]["gemini_direct"] = "not configured"
    
    # Check Gemini library status
    if response_generator.gemini_model:
        status["components"]["gemini_library"] = "available"
    else:
        status["components"]["gemini_library"] = "unavailable"
    
    # Check OpenAI status
    if response_generator.openai_client:
        status["components"]["openai"] = "available"
    else:
        status["components"]["openai"] = "unavailable"
    
    return jsonify(status)

asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(port=Config.PORT, debug=False)