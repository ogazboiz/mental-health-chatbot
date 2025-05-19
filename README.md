# NeuralEase: Mental Health Support Chatbot

NeuralEase is an AI-powered mental health support chatbot designed to provide compassionate and informative responses to users' mental health concerns. It uses advanced NLP techniques and a cascading fallback system to ensure robust, reliable, and helpful interactions.

![NeuralEase](https://i.imgur.com/Ndf3cXI.png)

## Features

- **User Authentication**: Secure login and registration system
- **Multiple Chat Sessions**: Users can maintain and manage multiple conversations
- **Enhanced Message Management**: Edit and delete messages
- **Emotional Intelligence**: Advanced emotion and sentiment detection
- **Intent Classification**: Accurately identifies user intents
- **Crisis Detection**: Identifies potential crisis situations and provides appropriate resources
- **Cascading Fallback System**: Uses multiple AI models to ensure reliable responses
- **Swagger API Documentation**: Well-documented API endpoints
- **Privacy-Focused**: Robust consent management for user data

## API Endpoints

### Authentication
- `POST /api/auth/register`: Register a new user
- `POST /api/auth/login`: Login a user
- `POST /api/auth/logout`: Logout a user
- `POST /api/auth/refresh`: Refresh authentication token

### User Profile
- `GET /api/user/profile`: Get user profile
- `PUT /api/user/profile`: Update user profile

### Chat Sessions
- `GET /api/sessions`: Get all user chat sessions
- `POST /api/sessions`: Create a new chat session
- `GET /api/sessions/{session_id}`: Get a specific chat session
- `PUT /api/sessions/{session_id}`: Update a chat session
- `DELETE /api/sessions/{session_id}`: Delete a chat session

### Messages
- `POST /api/sessions/{session_id}/messages`: Send a message
- `PUT /api/sessions/{session_id}/messages/{message_id}`: Edit a message
- `DELETE /api/sessions/{session_id}/messages/{message_id}`: Delete a message

### Legacy Endpoints
- `POST /consent`: Set consent for data storage
- `POST /feedback`: Submit feedback
- `POST /chat`: Legacy chat endpoint
- `GET /status`: Get API status

## Technology Stack

- **Frontend**: Not included in this repository
- **Backend**: Python with Flask/Flask-RestX
- **AI Models**:
  - Google's Gemini API
  - OpenAI API (fallback)
  - Hugging Face models for sentiment and emotion analysis
- **Data Storage**: File-based encrypted storage
- **Authentication**: Custom JWT-like token system

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/neural-ease.git
   cd neural-ease
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file with your API keys and configuration:
   ```
   # API Keys
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   HF_API_KEY=your_huggingface_api_key_here
   
   # Security
   ENCRYPTION_KEY=gKrIjy-esAkcFlwKR3z73gsCcxWOSaRMQzrHDkCVOL0=
   TOKEN_EXPIRY_HOURS=24
   SECURE_COOKIES=FALSE
   
   # Server Settings
   PORT=5000
   ```

6. Create necessary directories:
   ```bash
   mkdir -p sessions users
   ```

## Running the Application

### Development Server
```bash
python fixed_swagger_app.py
```

Access the API at http://localhost:5000 and Swagger documentation at http://localhost:5000/swagger

### Production Deployment
```bash
gunicorn wsgi:app
```

## Deployment Options

### Docker
```bash
docker-compose up -d
```

### Render
The application includes a `render.yaml` file for easy deployment to Render.

1. Push your code to GitHub
2. Create a new Render Blueprint
3. Connect to your GitHub repository
4. Add your environment variables in the Render dashboard

## API Documentation

Full API documentation is available through Swagger UI at `/swagger` when the application is running. This provides an interactive interface to explore and test all endpoints.

## Architecture

### Components

1. **User Authentication**: Manages user registration, login, and token validation
2. **Conversation Management**: Handles chat sessions and message history
3. **NLP Processing**: Analyzes user input for intent, sentiment, and emotions
4. **Response Generation**: Uses a cascading system:
   - Primary: Gemini API
   - Secondary: OpenAI API
   - Tertiary: Built-in responses

### Data Flow

1. User sends a message through an endpoint
2. Message is analyzed for safety and mental health relevance
3. NLP processing extracts intent, sentiment, and emotions
4. Response generator creates an appropriate reply
5. Conversation history is updated and encrypted
6. Response is returned to the user

## Security Considerations

- All user data is encrypted using Fernet symmetric encryption
- Passwords are hashed with SHA-256 + salt
- Authentication uses secure tokens with expiration
- Environment variables for sensitive configuration
- Explicit user consent required for data storage

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Google's Gemini API
- Uses Hugging Face transformer models for NLP
- Inspired by advances in AI for mental health support

## Disclaimer

This chatbot is designed to provide information and support related to mental health topics. It is not a substitute for professional medical advice, diagnosis, or treatment. If you or someone you know is experiencing a mental health crisis, please contact a professional healthcare provider or emergency services immediately.

---

For any questions or support, please open an issue in the GitHub repository.
