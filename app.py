import asyncio
import os
import time
from flask import Flask, request, jsonify, make_response, render_template_string
from flask_cors import CORS, cross_origin
from modules.conversation import Conversation
from modules.mental_health_response_generator import MentalHealthResponseGenerator  
from modules.nlp_processor import NLPProcessor
from modules.safety_checker import SafetyChecker
from modules.mental_health_filter import MentalHealthFilter
from modules.user_auth import User, AuthToken
from config import Config
import logging
from datetime import datetime

from asgiref.wsgi import WsgiToAsgi
from asgiref.sync import async_to_sync
from functools import wraps

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

# ============================================================================
# SIMPLIFIED CORS CONFIGURATION
# ============================================================================

# Get environment variables for CORS origins
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://uyi-mental-health-v1.vercel.app')
RENDER_URL = os.environ.get('RENDER_EXTERNAL_URL', '')

# Build origins list
cors_origins = [
    # Local development
    "http://localhost:3000",
    "http://localhost:8080", 
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    # Production
    "https://uyi-mental-health-v1.vercel.app",
    FRONTEND_URL,
]

# Add Render URL if available
if RENDER_URL:
    cors_origins.append(RENDER_URL)
    cors_origins.append(RENDER_URL.rstrip('/'))

# Clean up origins list
cors_origins = list(set(filter(None, cors_origins)))

print(f"🔗 CORS Origins configured: {cors_origins}")

# Enable CORS for all routes with Flask-CORS extension
CORS(app, 
     origins=cors_origins,
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     supports_credentials=True,
     automatic_options=True)

# Enable Flask-CORS debug logging
logging.getLogger('flask_cors').level = logging.DEBUG

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

# ============================================================================
# DOCUMENTATION TEMPLATE
# ============================================================================

DOCS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuralEase API Documentation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; color: #333; background: #f8f9fa;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px 0; text-align: center; margin-bottom: 30px;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .section { background: white; margin: 20px 0; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .section h2 { color: #667eea; margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .section h3 { color: #555; margin: 20px 0 10px 0; }
        .endpoint { 
            background: #f8f9fa; border-left: 4px solid #667eea; padding: 15px; margin: 10px 0; border-radius: 5px;
        }
        .method { 
            display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em; margin-right: 10px;
        }
        .get { background: #28a745; color: white; }
        .post { background: #007bff; color: white; }
        .put { background: #ffc107; color: #333; }
        .delete { background: #dc3545; color: white; }
        .crisis-box { 
            background: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 8px; margin: 20px 0;
        }
        .crisis-box h3 { color: #856404; margin-bottom: 10px; }
        .crisis-box strong { color: #d63384; }
        .quick-start { background: #e7f3ff; border-left: 4px solid #007bff; padding: 20px; margin: 20px 0; }
        .auth-box { background: #f0f8f0; border-left: 4px solid #28a745; padding: 20px; margin: 20px 0; }
        .code { background: #f4f4f4; padding: 10px; border-radius: 4px; font-family: monospace; margin: 10px 0; }
        .example { background: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; border-radius: 4px; margin: 10px 0; }
        .badge { background: #667eea; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .nav { background: white; padding: 15px 0; border-bottom: 1px solid #eee; position: sticky; top: 0; z-index: 100; }
        .nav ul { list-style: none; display: flex; justify-content: center; flex-wrap: wrap; }
        .nav li { margin: 0 15px; }
        .nav a { text-decoration: none; color: #667eea; font-weight: 500; padding: 5px 10px; border-radius: 5px; }
        .nav a:hover { background: #f0f0f0; }
        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header h1 { font-size: 2em; }
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>🧠 NeuralEase API</h1>
            <p>Mental Health Chatbot API with Authentication & Session Management</p>
            <div style="margin-top: 20px;">
                <span class="badge">v1.0.0</span>
                <span class="badge">REST API</span>
                <span class="badge">JWT Auth</span>
            </div>
        </div>
    </div>

    <nav class="nav">
        <div class="container">
            <ul>
                <li><a href="#quick-start">🚀 Quick Start</a></li>
                <li><a href="#auth">🔐 Authentication</a></li>
                <li><a href="#sessions">💬 Sessions</a></li>
                <li><a href="#messages">📝 Messages</a></li>
                <li><a href="#legacy">🔄 Legacy</a></li>
                <li><a href="#crisis">🆘 Crisis</a></li>
            </ul>
        </div>
    </nav>

    <div class="container">
        <div class="crisis-box">
            <h3>🆘 Mental Health Crisis Resources</h3>
            <p><strong>If you're experiencing a mental health crisis, please reach out immediately:</strong></p>
            <ul style="margin: 10px 0 0 20px;">
                <li><strong>US Suicide & Crisis Lifeline:</strong> 988</li>
                <li><strong>Crisis Text Line:</strong> Text HOME to 741741</li>
                <li><strong>Emergency Services:</strong> 911</li>
                <li><strong>International:</strong> Visit findahelpline.com</li>
            </ul>
        </div>

        <section id="quick-start" class="section">
            <h2>🚀 Quick Start</h2>
            <div class="quick-start">
                <h3>Get Started in 4 Steps:</h3>
                <ol style="margin: 10px 0 0 20px;">
                    <li><strong>Register:</strong> Create account at <code>POST /api/auth/register</code></li>
                    <li><strong>Login:</strong> Get your JWT token at <code>POST /api/auth/login</code></li>
                    <li><strong>Create Session:</strong> Start chatting at <code>POST /api/sessions</code></li>
                    <li><strong>Send Messages:</strong> Chat at <code>POST /api/sessions/{session_id}/messages</code></li>
                </ol>
            </div>
            
            <h3>Base URL</h3>
            <div class="code">{{ base_url }}</div>
            
            <h3>System Status</h3>
            <div class="grid">
                <div class="endpoint">
                    <span class="method get">GET</span><strong>/health</strong><br>
                    Basic health check
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span><strong>/status</strong><br>
                    Detailed system status
                </div>
            </div>
        </section>

        <section id="auth" class="section">
            <h2>🔐 Authentication</h2>
            <div class="auth-box">
                <h3>🔑 Authorization Header</h3>
                <p>Include your JWT token in requests:</p>
                <div class="code">Authorization: Bearer &lt;your_jwt_token&gt;</div>
            </div>

            <h3>Authentication Endpoints</h3>
            
            <div class="endpoint">
                <span class="method post">POST</span><strong>/api/auth/register</strong><br>
                Register a new user account
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "username": "john_doe", "email": "john@example.com", "password": "securepass123" }</pre>
                </div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span><strong>/api/auth/login</strong><br>
                Login with username and password
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "username": "john_doe", "password": "securepass123" }</pre>
                </div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span><strong>/api/auth/logout</strong><br>
                Logout user (clears token cookie)
            </div>

            <div class="endpoint">
                <span class="method post">POST</span><strong>/api/auth/refresh</strong><br>
                Refresh authentication token <span class="badge">Requires Auth</span>
            </div>
        </section>

        <section class="section">
            <h2>👤 User Profile</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span><strong>/api/user/profile</strong> <span class="badge">Requires Auth</span><br>
                Get user profile information
            </div>

            <div class="endpoint">
                <span class="method put">PUT</span><strong>/api/user/profile</strong> <span class="badge">Requires Auth</span><br>
                Update user profile
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "profile": { "preferences": { "theme": "dark" }, "bio": "Mental health advocate" } }</pre>
                </div>
            </div>
        </section>

        <section id="sessions" class="section">
            <h2>💬 Chat Sessions</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span><strong>/api/sessions</strong> <span class="badge">Requires Auth</span><br>
                Get all chat sessions for user
            </div>

            <div class="endpoint">
                <span class="method post">POST</span><strong>/api/sessions</strong> <span class="badge">Requires Auth</span><br>
                Create a new chat session
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "title": "Daily Check-in" }</pre>
                </div>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span><strong>/api/sessions/{session_id}</strong> <span class="badge">Requires Auth</span><br>
                Get specific session with full message history
            </div>

            <div class="endpoint">
                <span class="method put">PUT</span><strong>/api/sessions/{session_id}</strong> <span class="badge">Requires Auth</span><br>
                Update session (rename)
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "title": "Anxiety Support Session" }</pre>
                </div>
            </div>

            <div class="endpoint">
                <span class="method delete">DELETE</span><strong>/api/sessions/{session_id}</strong> <span class="badge">Requires Auth</span><br>
                Delete a chat session
            </div>
        </section>

        <section id="messages" class="section">
            <h2>📝 Messages</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span><strong>/api/sessions/{session_id}/messages</strong> <span class="badge">Requires Auth</span><br>
                Send a message to the chatbot
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "message": "I am feeling anxious about work today" }</pre>
                    <strong>Response includes:</strong>
                    <ul style="margin-top: 10px;">
                        <li>• AI-generated response</li>
                        <li>• Sentiment analysis</li>
                        <li>• Intent detection</li>
                        <li>• Safety filtering</li>
                        <li>• Crisis detection</li>
                    </ul>
                </div>
            </div>

            <div class="endpoint">
                <span class="method put">PUT</span><strong>/api/sessions/{session_id}/messages/{message_id}</strong> <span class="badge">Requires Auth</span><br>
                Edit a message in the conversation
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "content": "I am feeling better now" }</pre>
                </div>
            </div>

            <div class="endpoint">
                <span class="method delete">DELETE</span><strong>/api/sessions/{session_id}/messages/{message_id}</strong> <span class="badge">Requires Auth</span><br>
                Delete a message from the conversation
            </div>
        </section>

        <section id="legacy" class="section">
            <h2>🔄 Legacy Endpoints</h2>
            <p style="margin-bottom: 20px;"><em>These endpoints are maintained for backward compatibility.</em></p>
            
            <div class="endpoint">
                <span class="method post">POST</span><strong>/chat</strong><br>
                Legacy chat endpoint - simple message/response
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "message": "Hello, I need someone to talk to", "session_id": "optional" }</pre>
                </div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span><strong>/consent</strong><br>
                Set user consent for data storage
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "session_id": "session_uuid", "consent": true }</pre>
                </div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span><strong>/feedback</strong><br>
                Submit feedback about a chat session
                <div class="example">
                    <strong>Request:</strong>
                    <pre>{ "session_id": "session_uuid", "satisfaction": 4, "comments": "Very helpful!" }</pre>
                </div>
            </div>
        </section>

        <section class="section">
            <h2>🧠 AI Features</h2>
            <div class="grid">
                <div class="endpoint">
                    <h3>🎯 Intent Detection</h3>
                    Automatically identifies the purpose of user messages (greeting, emotional support, crisis, etc.)
                </div>
                <div class="endpoint">
                    <h3>😊 Sentiment Analysis</h3>
                    Analyzes emotional tone of messages (positive, negative, neutral) to tailor responses
                </div>
                <div class="endpoint">
                    <h3>🛡️ Safety Filtering</h3>
                    Automatically detects and filters inappropriate content for user safety
                </div>
                <div class="endpoint">
                    <h3>🆘 Crisis Detection</h3>
                    Identifies potential crisis situations and provides immediate resources
                </div>
                <div class="endpoint">
                    <h3>🧭 Domain Filtering</h3>
                    Ensures conversations stay focused on mental health topics
                </div>
                <div class="endpoint">
                    <h3>💡 Context Awareness</h3>
                    Maintains conversation context for more natural, coherent responses
                </div>
            </div>
        </section>

        <section class="section">
            <h2>⚡ Response Codes</h2>
            <div class="grid">
                <div class="endpoint">
                    <span class="badge" style="background: #28a745;">200</span> <strong>Success</strong><br>
                    Request completed successfully
                </div>
                <div class="endpoint">
                    <span class="badge" style="background: #28a745;">201</span> <strong>Created</strong><br>
                    Resource created successfully
                </div>
                <div class="endpoint">
                    <span class="badge" style="background: #ffc107; color: #333;">400</span> <strong>Bad Request</strong><br>
                    Invalid request data
                </div>
                <div class="endpoint">
                    <span class="badge" style="background: #dc3545;">401</span> <strong>Unauthorized</strong><br>
                    Missing or invalid authentication
                </div>
                <div class="endpoint">
                    <span class="badge" style="background: #dc3545;">404</span> <strong>Not Found</strong><br>
                    Resource not found
                </div>
                <div class="endpoint">
                    <span class="badge" style="background: #dc3545;">500</span> <strong>Server Error</strong><br>
                    Internal server error
                </div>
            </div>
        </section>

        <section class="section" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h2 style="color: white; border-bottom: 2px solid rgba(255,255,255,0.3);">💙 Support Mental Health</h2>
            <p style="font-size: 1.1em; margin: 20px 0;">This API is designed to provide accessible mental health support through AI-powered conversations.</p>
            <p><strong>Remember:</strong> If you're in crisis, please reach out to professional help immediately.</p>
            <div style="margin-top: 20px;">
                <span class="badge" style="background: rgba(255,255,255,0.2);">Confidential</span>
                <span class="badge" style="background: rgba(255,255,255,0.2);">Supportive</span>
                <span class="badge" style="background: rgba(255,255,255,0.2);">Available 24/7</span>
            </div>
        </section>
    </div>
</body>
</html>
"""

# ============================================================================
# ROOT AND DOCUMENTATION ENDPOINTS
# ============================================================================

@app.route('/')
@cross_origin()
def index():
    """API Root - JSON response with navigation"""
    return jsonify({
        "name": "NeuralEase Mental Health Chatbot API",
        "version": "1.0.0",
        "status": "operational",
        "description": "A mental health support chatbot API with user authentication and session management",
        "documentation": {
            "interactive_docs": "/docs",
            "health_check": "/health",
            "api_status": "/status"
        },
        "endpoints": {
            "authentication": {
                "register": "POST /api/auth/register",
                "login": "POST /api/auth/login", 
                "logout": "POST /api/auth/logout",
                "refresh_token": "POST /api/auth/refresh"
            },
            "user_profile": {
                "get_profile": "GET /api/user/profile",
                "update_profile": "PUT /api/user/profile"
            },
            "sessions": {
                "get_all_sessions": "GET /api/sessions",
                "create_session": "POST /api/sessions",
                "get_session": "GET /api/sessions/{session_id}",
                "update_session": "PUT /api/sessions/{session_id}",
                "delete_session": "DELETE /api/sessions/{session_id}"
            },
            "messaging": {
                "send_message": "POST /api/sessions/{session_id}/messages",
                "edit_message": "PUT /api/sessions/{session_id}/messages/{message_id}",
                "delete_message": "DELETE /api/sessions/{session_id}/messages/{message_id}"
            },
            "legacy": {
                "chat": "POST /chat",
                "consent": "POST /consent",
                "feedback": "POST /feedback"
            }
        },
        "quick_start": [
            "1. Register at /api/auth/register",
            "2. Login at /api/auth/login",
            "3. Create session at /api/sessions", 
            "4. Start chatting at /api/sessions/{session_id}/messages"
        ],
        "crisis_resources": {
            "us_suicide_lifeline": "988",
            "crisis_text_line": "Text HOME to 741741",
            "emergency": "911",
            "international": "findahelpline.com"
        }
    })

@app.route('/docs')
@cross_origin()
def documentation():
    """Interactive API Documentation"""
    base_url = request.url_root.rstrip('/')
    return render_template_string(DOCS_HTML, base_url=base_url)

# CORS TEST ENDPOINTS
@app.route('/cors-test', methods=['GET', 'OPTIONS'])
@cross_origin()
def cors_test():
    """Test endpoint to verify CORS configuration"""
    return jsonify({
        "message": "CORS is working!",
        "origin": request.headers.get('Origin'),
        "allowed_origins": cors_origins,
        "method": request.method,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/cors-config')
@cross_origin()
def cors_config():
    """Debug endpoint to see CORS configuration"""
    return jsonify({
        "allowed_origins": cors_origins,
        "frontend_url": FRONTEND_URL,
        "render_url": RENDER_URL,
        "request_origin": request.headers.get('Origin'),
        "environment": os.environ.get('FLASK_ENV', 'production')
    })

@app.route('/health')
@cross_origin()
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

@app.route('/status')
@cross_origin()
def get_status():
    """Get API status and component health"""
    status = {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": os.environ.get('FLASK_ENV', 'production'),
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

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
@cross_origin()
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
@cross_origin()
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
@cross_origin()
def logout():
    """Logout a user by clearing the token cookie"""
    response = make_response(jsonify({"message": "Logout successful"}))
    response.delete_cookie('token')
    return response

@app.route('/api/auth/refresh', methods=['POST'])
@cross_origin()
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

# ============================================================================
# USER PROFILE ENDPOINTS
# ============================================================================

@app.route('/api/user/profile', methods=['GET'])
@cross_origin()
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
@cross_origin()
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

# ============================================================================
# CHAT SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/sessions', methods=['GET'])
@cross_origin()
@token_required
def get_sessions(user):
    """Get all chat sessions for a user"""
    sessions = user.get_all_sessions()
    return jsonify({
        "sessions": sessions,
        "count": len(sessions)
    })

@app.route('/api/sessions', methods=['POST'])
@cross_origin()
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
@cross_origin()
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
@cross_origin()
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
@cross_origin()
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

# ============================================================================
# MESSAGE MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/sessions/<session_id>/messages', methods=['POST'])
@cross_origin()
@token_required
def send_message(user, session_id):
    """Send a message to a chat session"""
    # Use async_to_sync to run the async function properly
    return async_to_sync(_async_send_message)(user, session_id)

@app.route('/api/sessions/<session_id>/messages/<message_id>', methods=['PUT'])
@cross_origin()
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
@cross_origin()
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

# ============================================================================
# LEGACY ENDPOINTS (Backward Compatibility)
# ============================================================================

@app.route('/consent', methods=['POST'])
@cross_origin()
def set_consent():
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

@app.route('/feedback', methods=['POST'])
@cross_origin()
def submit_feedback():
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

@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    """Legacy chat endpoint for backward compatibility"""
    return async_to_sync(_async_chat)()

# ============================================================================
# ASYNC IMPLEMENTATION FUNCTIONS
# ============================================================================

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

# Create ASGI app for deployment
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 5000))  # Default for local development
    
    print(f"🚀 Starting Mental Health Chatbot on port {port}")
    print(f"💚 Health check available at: http://localhost:{port}/health")
    print(f"📊 Status check available at: http://localhost:{port}/status")
    print(f"📋 API Documentation available at: http://localhost:{port}/docs")
    print(f"🌐 API Root: http://localhost:{port}/")
    print(f"🔧 CORS Test: http://localhost:{port}/cors-test")
    print(f"🔧 CORS Config: http://localhost:{port}/cors-config")
    print(f"")
    print(f"🎯 Quick Test URLs:")
    print(f"   - API Info: http://localhost:{port}/")
    print(f"   - Documentation: http://localhost:{port}/docs")
    print(f"   - Health: http://localhost:{port}/health")
    
    # Bind to 0.0.0.0 for Render (not just localhost)
    app.run(host='0.0.0.0', port=port, debug=False)
