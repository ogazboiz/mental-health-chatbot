import os
from flask import Flask, Blueprint
from flask_restx import Api, Resource, fields, Namespace
from config import Config

# Create a blueprint for the API documentation
api_bp = Blueprint('api', __name__)

# Initialize Flask-RestX API with the blueprint
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
# Authentication models
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

# Profile models
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

profile_response = api.model('ProfileResponse', {
    'user_id': fields.String(description='User ID'),
    'username': fields.String(description='Username'),
    'email': fields.String(description='Email address'),
    'profile': fields.Nested(profile_data),
    'created_at': fields.Float(description='Account creation timestamp'),
    'last_login': fields.Float(description='Last login timestamp')
})

# Session models
session_create = api.model('SessionCreate', {
    'title': fields.String(description='Chat session title')
})

session_update = api.model('SessionUpdate', {
    'title': fields.String(required=True, description='New chat session title')
})

session_response = api.model('SessionResponse', {
    'message': fields.String(description='Status message'),
    'session_id': fields.String(description='Session ID'),
    'title': fields.String(description='Chat session title')
})

sessions_list_response = api.model('SessionsListResponse', {
    'sessions': fields.List(fields.Raw, description='List of chat sessions'),
    'count': fields.Integer(description='Number of chat sessions')
})

message_send = api.model('MessageSend', {
    'message': fields.String(required=True, description='Message content')
})

message_edit = api.model('MessageEdit', {
    'content': fields.String(required=True, description='New message content')
})

message_response = api.model('MessageResponse', {
    'response': fields.String(description='Chatbot response'),
    'message_id': fields.String(description='Message ID'),
    'session_id': fields.String(description='Session ID'),
    'analysis': fields.Raw(description='Message analysis data'),
    'user_profile': fields.Raw(description='User profile data'),
    'source': fields.String(description='Response source'),
    'timestamp': fields.String(description='Response timestamp')
})

# Legacy models
consent_model = api.model('Consent', {
    'session_id': fields.String(description='Session ID'),
    'consent': fields.Boolean(required=True, description='Consent flag')
})

feedback_model = api.model('Feedback', {
    'session_id': fields.String(required=True, description='Session ID'),
    'satisfaction': fields.Integer(required=True, min=1, max=5, description='Satisfaction score (1-5)'),
    'comments': fields.String(description='Feedback comments')
})

chat_model = api.model('Chat', {
    'message': fields.String(required=True, description='User message'),
    'session_id': fields.String(description='Session ID')
})

def init_swagger(app):
    """Initialize Swagger documentation for the Flask app"""
    app.register_blueprint(api_bp)
    return api