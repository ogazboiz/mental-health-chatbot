import base64
import os
import json
import time
import hashlib
import secrets
from typing import Dict, List, Any, Optional
import logging
from cryptography.fernet import Fernet
from config import Config

class User:
    def __init__(self):
        """Initialize a new user object"""
        self.user_id = None
        self.username = None
        self.email = None
        self.password_hash = None
        self.salt = None
        self.created_at = None
        self.last_login = None
        self.sessions = []  # List of session IDs belonging to this user
        self.profile = {
            "name": "",
            "age": None,
            "preferred_responses": "neutral",
            "consent_given": False,
            "notification_preferences": {
                "email_notifications": False,
                "session_reminders": False
            },
            "theme": "light"
        }
        self.user_file = None
        
    def create_user(self, username: str, email: str, password: str) -> bool:
        """Create a new user with hashed password"""
        # Check if username already exists
        if self._username_exists(username):
            logging.warning(f"Username already exists: {username}")
            return False
            
        # Generate a unique user ID
        self.user_id = base64.urlsafe_b64encode(os.urandom(16)).decode('utf-8')
        self.username = username
        self.email = email
        
        # Generate salt and hash password
        self.salt = secrets.token_hex(16)
        self.password_hash = self._hash_password(password, self.salt)
        
        self.created_at = time.time()
        self.last_login = time.time()
        self.user_file = f"users/{self.user_id}.json"
        
        # Create users directory if it doesn't exist
        os.makedirs("users", exist_ok=True)
        
        # Save user data
        return self.save_user()
    
    def save_user(self) -> bool:
        """Save user data to encrypted storage"""
        os.makedirs("users", exist_ok=True)
        
        user_data = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "salt": self.salt,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "sessions": self.sessions,
            "profile": self.profile
        }
        
        # Create a cipher for encryption
        cipher = Fernet(Config.ENCRYPTION_KEY.encode('utf-8'))
        
        try:
            encrypted_data = cipher.encrypt(json.dumps(user_data).encode('utf-8'))
            with open(self.user_file, 'wb') as f:
                f.write(encrypted_data)
            logging.debug(f"Saved user data for {self.username}")
            return True
        except Exception as e:
            logging.error(f"Failed to save user data: {str(e)}")
            return False
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate a user with username and password"""
        # Load user data by username
        if not self.load_by_username(username):
            logging.warning(f"Authentication failed: User not found: {username}")
            return False
        
        # Check password
        hashed_password = self._hash_password(password, self.salt)
        if hashed_password != self.password_hash:
            logging.warning(f"Authentication failed: Invalid password for {username}")
            return False
        
        # Update last login time
        self.last_login = time.time()
        self.save_user()
        
        logging.info(f"User authenticated: {username}")
        return True
    
    def load_by_username(self, username: str) -> bool:
        """Load user data by username"""
        # Find user file by username
        for filename in os.listdir("users"):
            if filename.endswith(".json"):
                try:
                    file_path = os.path.join("users", filename)
                    with open(file_path, 'rb') as f:
                        encrypted_data = f.read()
                    
                    # Decrypt data
                    cipher = Fernet(Config.ENCRYPTION_KEY.encode('utf-8'))
                    decrypted_data = cipher.decrypt(encrypted_data).decode('utf-8')
                    user_data = json.loads(decrypted_data)
                    
                    if user_data["username"] == username:
                        # User found, load data
                        self.user_id = user_data["user_id"]
                        self.username = user_data["username"]
                        self.email = user_data["email"]
                        self.password_hash = user_data["password_hash"]
                        self.salt = user_data["salt"]
                        self.created_at = user_data["created_at"]
                        self.last_login = user_data["last_login"]
                        self.sessions = user_data["sessions"]
                        self.profile = user_data["profile"]
                        self.user_file = file_path
                        
                        logging.debug(f"Loaded user data for {username}")
                        return True
                except Exception as e:
                    logging.error(f"Error while loading user data: {str(e)}")
                    continue
        
        logging.warning(f"User not found: {username}")
        return False
    
    def load_by_user_id(self, user_id: str) -> bool:
        """Load user data by user ID"""
        file_path = f"users/{user_id}.json"
        if not os.path.exists(file_path):
            logging.warning(f"User ID not found: {user_id}")
            return False
        
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt data
            cipher = Fernet(Config.ENCRYPTION_KEY.encode('utf-8'))
            decrypted_data = cipher.decrypt(encrypted_data).decode('utf-8')
            user_data = json.loads(decrypted_data)
            
            # Load data
            self.user_id = user_data["user_id"]
            self.username = user_data["username"]
            self.email = user_data["email"]
            self.password_hash = user_data["password_hash"]
            self.salt = user_data["salt"]
            self.created_at = user_data["created_at"]
            self.last_login = user_data["last_login"]
            self.sessions = user_data["sessions"]
            self.profile = user_data["profile"]
            self.user_file = file_path
            
            logging.debug(f"Loaded user data for ID {user_id}")
            return True
        except Exception as e:
            logging.error(f"Error while loading user data: {str(e)}")
            return False
    
    def update_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Update user profile data"""
        # Update profile with new data
        for key, value in profile_data.items():
            if key in self.profile:
                self.profile[key] = value
                
        return self.save_user()
    
    def add_session(self, session_id: str) -> bool:
        """Add a session ID to user's sessions"""
        if session_id not in self.sessions:
            self.sessions.append(session_id)
            return self.save_user()
        return True
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session ID from user's sessions"""
        if session_id in self.sessions:
            self.sessions.remove(session_id)
            return self.save_user()
        return True
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get metadata for all user sessions"""
        session_data = []
        
        for session_id in self.sessions:
            try:
                # Try to load basic session info without decrypting all messages
                file_path = f"sessions/{session_id}.json"
                if not os.path.exists(file_path):
                    continue
                
                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()
                
                # Decrypt data
                cipher = Fernet(Config.ENCRYPTION_KEY.encode('utf-8'))
                decrypted_data = cipher.decrypt(encrypted_data).decode('utf-8')
                session_info = json.loads(decrypted_data)
                
                # Extract minimal metadata
                last_message = "No messages"
                timestamp = session_info.get("last_interaction", 0)
                
                if session_info.get("messages") and len(session_info["messages"]) > 0:
                    last_msg = session_info["messages"][-1]
                    if last_msg.get("role") == "system":
                        last_message = last_msg.get("content", "")[:50] + "..."
                
                session_data.append({
                    "session_id": session_id,
                    "last_activity": timestamp,
                    "last_message": last_message,
                    "message_count": len(session_info.get("messages", []))
                })
                
            except Exception as e:
                logging.error(f"Error loading session {session_id}: {str(e)}")
                continue
        
        # Sort by last activity (newest first)
        session_data.sort(key=lambda x: x["last_activity"], reverse=True)
        return session_data
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash a password with salt using SHA-256"""
        salted_password = password + salt
        return hashlib.sha256(salted_password.encode('utf-8')).hexdigest()
    
    def _username_exists(self, username: str) -> bool:
        """Check if a username already exists"""
        # Create users directory if it doesn't exist
        os.makedirs("users", exist_ok=True)
        
        for filename in os.listdir("users"):
            if filename.endswith(".json"):
                try:
                    file_path = os.path.join("users", filename)
                    with open(file_path, 'rb') as f:
                        encrypted_data = f.read()
                    
                    # Decrypt data
                    cipher = Fernet(Config.ENCRYPTION_KEY.encode('utf-8'))
                    decrypted_data = cipher.decrypt(encrypted_data).decode('utf-8')
                    user_data = json.loads(decrypted_data)
                    
                    if user_data["username"] == username:
                        return True
                except Exception as e:
                    logging.error(f"Error while checking username: {str(e)}")
                    continue
        
        return False

class AuthToken:
    """Simple JWT-like token management for user authentication"""
    
    @staticmethod
    def generate_token(user_id: str) -> str:
        """Generate a token for a user"""
        # Create a payload with user ID and expiration time
        payload = {
            "user_id": user_id,
            "exp": time.time() + (Config.TOKEN_EXPIRY_HOURS * 3600)
        }
        
        # Encode and encrypt the payload
        json_payload = json.dumps(payload)
        cipher = Fernet(Config.ENCRYPTION_KEY.encode('utf-8'))
        encrypted_payload = cipher.encrypt(json_payload.encode('utf-8'))
        
        # Return the token
        return base64.urlsafe_b64encode(encrypted_payload).decode('utf-8')
    
    @staticmethod
    def validate_token(token: str) -> Optional[str]:
        """Validate a token and return the user ID if valid"""
        try:
            # Decode and decrypt the token
            encrypted_payload = base64.urlsafe_b64decode(token.encode('utf-8'))
            cipher = Fernet(Config.ENCRYPTION_KEY.encode('utf-8'))
            decrypted_payload = cipher.decrypt(encrypted_payload).decode('utf-8')
            payload = json.loads(decrypted_payload)
            
            # Check expiration
            if payload["exp"] < time.time():
                logging.warning("Token expired")
                return None
            
            # Return user ID
            return payload["user_id"]
        except Exception as e:
            logging.error(f"Token validation error: {str(e)}")
            return None