import base64
import os
import json
import time
from cryptography.fernet import Fernet
from typing import Dict, List, Any, Optional
import logging
from config import Config

class Conversation:
    def __init__(self, encryption_key: str):
        """Initialize a new conversation with a unique session ID"""
        self.session_id = base64.urlsafe_b64encode(os.urandom(16)).decode('utf-8')
        self.encryption_key = encryption_key.encode('utf-8')
        self.cipher = Fernet(self.encryption_key)
        self.messages: List[Dict[str, Any]] = []
        self.user_id = None  # Owner of this conversation
        self.title = "New Conversation"  # Default title
        self.created_at = time.time()
        self.user_profile = {
            "consent_given": False,
            "name": "",
            "age": None,
            "preferred_responses": "neutral",
            "emotion_history": [],
            "sentiment_history": [],
            "primary_concerns": []
        }
        self.last_interaction = time.time()
        self.session_file = f"sessions/{self.session_id}.json"
        self.deleted = False

    def set_user_id(self, user_id: str):
        """Associate this conversation with a user"""
        self.user_id = user_id
        logging.debug(f"Conversation {self.session_id} associated with user {user_id}")
        
    def set_title(self, title: str):
        """Set the conversation title"""
        self.title = title
        logging.debug(f"Conversation {self.session_id} title set to '{title}'")
        self.save_session()

    def set_consent(self, consent: bool):
        """Set user consent for data storage"""
        self.user_profile["consent_given"] = consent
        logging.debug(f"Consent set to {consent} for session {self.session_id}")

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]]):
        """Add a message to the conversation history"""
        if len(self.messages) >= Config.MAX_CONVERSATION_LENGTH:
            self.messages.pop(0)
        message = {
            "id": base64.urlsafe_b64encode(os.urandom(8)).decode('utf-8'),  # Unique ID for message
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "edited": False,
            "edit_history": []  # Track edits for transparency
        }
        self.messages.append(message)
        if role == "user" and metadata:
            # Update user profile based on message metadata
            if metadata.get('emotions') and metadata.get('emotions') != 'none':
                self.user_profile["emotion_history"].append(metadata['emotions'])
                # Keep history at a reasonable size
                if len(self.user_profile["emotion_history"]) > 10:
                    self.user_profile["emotion_history"].pop(0)
                    
            if metadata.get('sentiment', {}).get('label'):
                self.user_profile["sentiment_history"].append(metadata['sentiment']['label'])
                # Keep history at a reasonable size
                if len(self.user_profile["sentiment_history"]) > 10:
                    self.user_profile["sentiment_history"].pop(0)
                    
            # Extract potential primary concerns
            if metadata.get('keywords'):
                for keyword in metadata['keywords']:
                    if keyword.lower() not in [concern.lower() for concern in self.user_profile["primary_concerns"]]:
                        if len(self.user_profile["primary_concerns"]) >= 5:
                            self.user_profile["primary_concerns"].pop(0)
                        self.user_profile["primary_concerns"].append(keyword)
                        
        self.last_interaction = time.time()
        
        # Auto-generate title from the first user message if not set
        if role == "user" and len(self.messages) <= 2 and self.title == "New Conversation":
            # Create a simple title from the first few words
            words = content.split()
            if len(words) > 2:
                self.title = " ".join(words[:4]) + "..."
            else:
                self.title = content
                
        logging.debug(f"Added {role} message to session {self.session_id}: {content[:50]}...")

    def edit_message(self, message_id: str, new_content: str) -> bool:
        """Edit an existing message"""
        for message in self.messages:
            if message.get("id") == message_id:
                # Store the original message in edit history
                if not message.get("edit_history"):
                    message["edit_history"] = []
                    
                message["edit_history"].append({
                    "content": message["content"],
                    "edited_at": time.time()
                })
                
                # Update the message
                message["content"] = new_content
                message["edited"] = True
                self.last_interaction = time.time()
                
                logging.debug(f"Edited message {message_id} in session {self.session_id}")
                return True
                
        logging.warning(f"Message {message_id} not found in session {self.session_id}")
        return False
    
    def delete_message(self, message_id: str) -> bool:
        """Delete a message by ID"""
        for i, message in enumerate(self.messages):
            if message.get("id") == message_id:
                # Instead of completely removing, mark as deleted
                self.messages[i]["content"] = "[Message deleted]"
                self.messages[i]["deleted"] = True
                self.last_interaction = time.time()
                
                logging.debug(f"Deleted message {message_id} in session {self.session_id}")
                return True
                
        logging.warning(f"Message {message_id} not found in session {self.session_id}")
        return False

    def get_context(self) -> List[Dict[str, Any]]:
        """Get recent conversation context for response generation"""
        return self.messages[-Config.CONTEXT_WINDOW:]

    def get_user_profile(self) -> Dict[str, Any]:
        """Get the current user profile"""
        return self.user_profile

    def save_session(self):
        """Save the session to encrypted storage if consent is given"""
        if not self.user_profile["consent_given"]:
            logging.debug(f"Session {self.session_id} not saved (no consent)")
            return
            
        os.makedirs("sessions", exist_ok=True)
        session_data = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at,
            "messages": self.messages,
            "user_profile": self.user_profile,
            "last_interaction": self.last_interaction,
            "deleted": self.deleted
        }
        
        try:
            encrypted_data = self.cipher.encrypt(json.dumps(session_data).encode('utf-8'))
            with open(self.session_file, 'wb') as f:
                f.write(encrypted_data)
            logging.debug(f"Saved session {self.session_id}")
        except Exception as e:
            logging.error(f"Failed to save session {self.session_id}: {str(e)}")

    def load_session(self, session_id: str) -> bool:
        """Load a session from storage by ID"""
        self.session_id = session_id
        self.session_file = f"sessions/{session_id}.json"
        
        if not os.path.exists(self.session_file):
            logging.warning(f"Session file not found: {self.session_file}")
            return False
            
        try:
            with open(self.session_file, 'rb') as f:
                encrypted_data = f.read()
                
            decrypted_data = self.cipher.decrypt(encrypted_data).decode('utf-8')
            session_data = json.loads(decrypted_data)
            
            # Load session data
            self.messages = session_data["messages"]
            self.user_profile = session_data["user_profile"]
            self.last_interaction = session_data["last_interaction"]
            self.user_id = session_data.get("user_id")
            self.title = session_data.get("title", "Conversation")
            self.created_at = session_data.get("created_at", time.time())
            self.deleted = session_data.get("deleted", False)
            
            # Check for session expiry
            if time.time() - self.last_interaction > Config.SESSION_EXPIRY_MINUTES * 60:
                logging.warning(f"Session {session_id} has expired")
                return False
                
            logging.debug(f"Loaded session {session_id} with {len(self.messages)} messages")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load session {session_id}: {str(e)}")
            return False
    
    def mark_deleted(self):
        """Mark the conversation as deleted (soft delete)"""
        self.deleted = True
        self.save_session()
        logging.info(f"Marked session {self.session_id} as deleted")
            
    def update_user_preference(self, preference_type: str, value: Any) -> bool:
        """Update a user preference in the profile"""
        if preference_type in self.user_profile:
            self.user_profile[preference_type] = value
            logging.debug(f"Updated {preference_type} to {value} for session {self.session_id}")
            return True
        return False