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
from flask_restx import Api, Resource, fields, Namespace, Blueprint

# Setup NLTK first before other imports
print("Setting up NLTK data...")
try:
    import setup_nltk
    setup_nltk.setup_nltk_data()
    print("✅ NLTK setup completed")
except Exception as e:
    print(f"⚠️ NLTK setup failed: {e}")

# Enforce WindowsSelectorEventLoopPolicy before any asyncio operations
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Also log to console for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger('').addHandler(console_handler)

app = Flask(__name__)

# Initialize Swagger documentation
api_bp = Blueprint('api', __name__)

authorizations = {
    'Bearer': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization',
        'description': 'Enter: **Bearer &lt;token&gt;**'
    }
}

api = Api(
    api_bp,
    version='1.0',
    title='Mental Health Chatbot API',
    description='API documentation for the Mental Health Chatbot',
    doc='/swagger',
    authorizations=authorizations,
    security='Bearer'
)

# Create namespaces for different endpoint groups
auth_ns = api.namespace('api/auth', description='Authentication operations')
user_ns = api.namespace('api/user', description='User profile operations')
sessions_ns = api.namespace('api/sessions', description='Chat session operations')
legacy_ns = api.namespace('legacy', description='Legacy endpoints')

# Define models for request/response schemas
register_model = api.model('Register', {
    'username': fields.String(required=True, description='Username'),
    'email': fields.String(required=True, description='Email address'),
    'password': fields.String(required=True, description='Password')
})

login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password')
})

auth_response = api.model('AuthResponse', {
    'message': fields.String(description='Status message'),
    'user_id': fields.String(description='User ID'),
    'username': fields.String(description='Username'),
    'email': fields.String(description='Email address'),
    'token': fields.String(description='Authentication token')
})

profile_data = api.model('ProfileData', {
    'name': fields.String(description='User name'),
    'age': fields.Integer(description='User age'),
    'preferred_responses': fields.String(description='Response style preference'),
    'theme': fields.String(description='UI theme preference'),
    'notification_preferences': fields.Raw(description='Notification settings')
})

profile_update = api.model('ProfileUpdate', {
    'profile': fields.Nested(profile_data)
})

session_create = api.model('SessionCreate', {
    'title': fields.String(description='Chat session title')
})

message_send = api.model('MessageSend', {
    'message': fields.String(required=True, description='Message content')
})

chat_model = api.model('Chat', {
    'message': fields.String(required=True, description='User message'),
    'session_id': fields.String(description='Session ID')
})

consent_model = api.model('Consent', {
    'session_id': fields.String(description='Session ID'),
    'consent': fields.Boolean(required=True, description='Consent flag')
})

feedback_model = api.model('Feedback', {
    'session_id': fields.String(required=True, description='Session ID'),
    'satisfaction': fields.Integer(required=True, min=1, max=5, description='Satisfaction score (1-5)'),
    'comments': fields.String(description='Feedback comments')
})

# Register the blueprint
app.register_blueprint(api_bp)

# Initialize components
print("Initializing chatbot components...")
try:
    nlp_processor = NLPProcessor()
    response_generator = MentalHealthResponseGenerator()
    safety_checker = SafetyChecker()
    mental_health_filter = MentalHealthFilter()
    print("✅ All components initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize components: {e}")
    # Create simple fallback components
    nlp_processor = None
    response_generator = None
    safety_checker = None
    mental_health_filter = None

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
@auth_ns.route('/register')
class Register(Resource):
    @auth_ns.doc('register_user')
    @auth_ns.expect(register_model)
    @auth_ns.response(200, 'Success', auth_response)
    @auth_ns.response(400, 'Validation Error')
    @auth_ns.response(409, 'Username already exists')
    def post(self):
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

@auth_ns.route('/login')
class Login(Resource):
    @auth_ns.doc('login_user')
    @auth_ns.expect(login_model)
    @auth_ns.response(200, 'Success', auth_response)
    @auth_ns.response(401, 'Invalid credentials')
    def post(self):
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

@auth_ns.route('/logout')
class Logout(Resource):
    @auth_ns.doc('logout_user')
    @auth_ns.response(200, 'Success')
    def post(self):
        """Logout a user by clearing the token cookie"""
        response = make_response(jsonify({"message": "Logout successful"}))
        response.delete_cookie('token')
        return response

# User Profile Endpoints
@user_ns.route('/profile')
class UserProfile(Resource):
    @user_ns.doc('get_user_profile', security='Bearer')
    @user_ns.response(200, 'Success')
    @user_ns.response(401, 'Authentication Error')
    @token_required
    def get(self, user):
        """Get user profile"""
        return jsonify({
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "profile": user.profile,
            "created_at": user.created_at,
            "last_login": user.last_login
        })

# Chat Session Management Endpoints
@sessions_ns.route('/')
class Sessions(Resource):
    @sessions_ns.doc('get_sessions', security='Bearer')
    @sessions_ns.response(200, 'Success')
    @sessions_ns.response(401, 'Authentication Error')
    @token_required
    def get(self, user):
        """Get all chat sessions for a user"""
        sessions = user.get_all_sessions()
        return jsonify({
            "sessions": sessions,
            "count": len(sessions)
        })
    
    @sessions_ns.doc('create_session', security='Bearer')
    @sessions_ns.expect(session_create)
    @sessions_ns.response(200, 'Success')
    @sessions_ns.response(401, 'Authentication Error')
    @token_required
    def post(self, user):
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

@sessions_ns.route('/<string:session_id>/messages')
class SessionMessages(Resource):
    @sessions_ns.doc('send_message', security='Bearer')
    @sessions_ns.expect(message_send)
    @sessions_ns.response(200, 'Success')
    @sessions_ns.response(401, 'Authentication Error')
    @sessions_ns.response(404, 'Session not found')
    @token_required
    def post(self, user, session_id):
        """Send a message to a chat session"""
        # Use async_to_sync to run the async function properly
        return async_to_sync(_async_send_message)(user, session_id)

# Legacy Endpoints (for backward compatibility)
@legacy_ns.route('/consent')
class Consent(Resource):
    @legacy_ns.doc('set_consent')
    @legacy_ns.expect(consent_model)
    @legacy_ns.response(200, 'Success')
    @legacy_ns.response(400, 'Bad Request')
    def post(self):
        """Set user consent for data storage"""
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

@legacy_ns.route('/feedback')
class Feedback(Resource):
    @legacy_ns.doc('submit_feedback')
    @legacy_ns.expect(feedback_model)
    @legacy_ns.response(200, 'Success')
    @legacy_ns.response(400, 'Bad Request')
    def post(self):
        """Submit feedback about a chat session"""
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

@legacy_ns.route('/chat')
class LegacyChat(Resource):
    @legacy_ns.doc('chat')
    @legacy_ns.expect(chat_model)
    @legacy_ns.response(200, 'Success')
    @legacy_ns.response(400, 'Bad Request')
    @legacy_ns.response(403, 'Consent Required')
    def post(self):
        """Legacy chat endpoint for backward compatibility"""
        return async_to_sync(_async_chat)()

@legacy_ns.route('/status')
class Status(Resource):
    @legacy_ns.doc('get_status')
    @legacy_ns.response(200, 'Success')
    def get(self):
        """Get API status and component health"""
        status = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "nlp_processor": "available" if nlp_processor else "unavailable",
                "response_generator": "available" if response_generator else "unavailable",
                "safety_checker": "available" if safety_checker else "unavailable",
                "mental_health_filter": "available" if mental_health_filter else "unavailable"
            }
        }
        
        if response_generator:
            # Check AI API status
            if hasattr(response_generator, 'gemini_api_key') and response_generator.gemini_api_key:
                status["components"]["gemini_api"] = "configured"
            else:
                status["components"]["gemini_api"] = "not configured"
            
            if hasattr(response_generator, 'openai_client') and response_generator.openai_client:
                status["components"]["openai_api"] = "available"
            else:
                status["components"]["openai_api"] = "unavailable"
        
        return status

# Simple fallback response function
def get_fallback_response(intent, user_input):
    """Simple fallback when AI components are unavailable"""
    fallback_responses = {
        "greeting": "Hello! I'm NeuralEase, here to support you with mental health concerns. How are you feeling today?",
        "emotional_support": "I hear that you're going through a difficult time. Your feelings are valid, and you're not alone. Would you like to talk more about what you're experiencing?",
        "crisis": "I'm concerned about your wellbeing. If you're in crisis, please call 988 for immediate support from the Suicide & Crisis Lifeline. They're available 24/7.",
        "default": "I'm here to listen and support you with mental health concerns. Could you tell me a bit more about how you're feeling or what's on your mind?"
    }
    
    # Simple keyword-based intent detection
    user_lower = user_input.lower()
    if any(word in user_lower for word in ["hello", "hi", "hey"]):
        return fallback_responses["greeting"]
    elif any(word in user_lower for word in ["suicide", "kill myself", "end my life"]):
        return fallback_responses["crisis"]
    elif any(word in user_lower for word in ["sad", "depressed", "anxious", "stressed"]):
        return fallback_responses["emotional_support"]
    else:
        return fallback_responses["default"]

# Async implementation functions
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

        # Use fallback if components are not available
        if not all([nlp_processor, response_generator, safety_checker, mental_health_filter]):
            logging.warning("Using fallback mode - some components unavailable")
            response = get_fallback_response("default", user_input)
            
            # Add basic metadata
            analysis = {"intent": {"intent": "general"}, "sentiment": {"label": "neutral"}, "emotions": "none"}
            conversation.add_message("user", user_input, analysis)
            conversation.add_message("system", response, {"source": "fallback", "model": "builtin"})
            conversation.save_session()
            
            return jsonify({
                "response": response,
                "session_id": session_id,
                "source": "fallback",
                "timestamp": datetime.now().isoformat()
            })

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
        
        # Extract intent and other analysis
        intent_dict = analysis.get('intent', {})
        intent = intent_dict.get('intent', 'general')
        
        sentiment_dict = analysis.get('sentiment', {})
        sentiment = sentiment_dict.get('label', 'neutral')
        
        emotions = analysis.get('emotions', 'none')
        
        # Get conversation context
        context = conversation.get_context()
        user_profile = conversation.get_user_profile()
        user_profile["last_input"] = user_input

        # Add user message
        conversation.add_message("user", user_input, analysis)

        # Generate response
        async with response_generator:
            response = await response_generator.generate_response(intent, sentiment, emotions, context, user_profile)

        # Add system response
        system_metadata = {
            "intent": analysis['intent'],
            "source": response_generator._last_source,
            "model": response_generator._last_source
        }
        conversation.add_message("system", response, system_metadata)
        conversation.save_session()

        response_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Session {session_id}: Response generated in {response_time}s")

        return jsonify({
            "response": response,
            "message_id": conversation.messages[-1]["id"],
            "session_id": session_id,
            "analysis": analysis,
            "user_profile": user_profile,
            "source": response_generator._last_source,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Send message error: {str(e)}", exc_info=True)
        # Fallback response for errors
        response = "I'm having trouble processing that right now. How are you feeling today?"
        return jsonify({
            "response": response,
            "session_id": session_id,
            "source": "error_fallback",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

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

        # Use fallback if components are not available
        if not all([nlp_processor, response_generator, safety_checker, mental_health_filter]):
            logging.warning("Using fallback mode - some components unavailable")
            response = get_fallback_response("default", user_input)
            
            # Add basic metadata
            analysis = {"intent": {"intent": "general"}, "sentiment": {"label": "neutral"}, "emotions": "none"}
            conversation.add_message("user", user_input, analysis)
            conversation.add_message("system", response, {"source": "fallback", "model": "builtin"})
            conversation.save_session()
            
            return jsonify({
                "response": response,
                "session_id": session_id,
                "source": "fallback",
                "timestamp": datetime.now().isoformat()
            })

        # Full processing with all components
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
        
        # Extract analysis components
        intent_dict = analysis.get('intent', {})
        intent = intent_dict.get('intent', 'general')
        
        sentiment_dict = analysis.get('sentiment', {})
        sentiment = sentiment_dict.get('label', 'neutral')
        
        emotions = analysis.get('emotions', 'none')
        
        # Get context and user profile
        context = conversation.get_context()
        user_profile = conversation.get_user_profile()
        user_profile["last_input"] = user_input

        # Add user message
        conversation.add_message("user", user_input, analysis)

        # Generate response
        async with response_generator:
            response = await response_generator.generate_response(intent, sentiment, emotions, context, user_profile)

        # Add system response
        system_metadata = {
            "intent": analysis['intent'],
            "source": response_generator._last_source,
            "model": response_generator._last_source
        }
        conversation.add_message("system", response, system_metadata)
        conversation.save_session()

        response_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Session {session_id}: Response generated in {response_time}s")

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
        # Fallback response for errors
        response = "I'm having trouble processing that right now. How are you feeling today?"
        return jsonify({
            "response": response,
            "session_id": session_id or "",
            "source": "error_fallback",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Simple health check endpoint
@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "nlp_processor": "available" if nlp_processor else "unavailable",
            "response_generator": "available" if response_generator else "unavailable",
            "safety_checker": "available" if safety_checker else "unavailable",
            "mental_health_filter": "available" if mental_health_filter else "unavailable"
        }
    })

# Create ASGI app for deployment
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', Config.PORT))
    
    print(f"🚀 Starting Mental Health Chatbot on port {port}")
    print(f"📊 Swagger docs available at: http://localhost:{port}/swagger")
    print(f"💚 Health check at: http://localhost:{port}/health")
    
    # Bind to 0.0.0.0 for Render (not just localhost)
    app.run(host='0.0.0.0', port=port, debug=False)