import requests
import json
import time
from typing import List, Dict, Any
import logging

logging.basicConfig(
    filename='test_chatbot.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_URL = "http://127.0.0.1:5000"

def send_consent(session_id: str, consent: bool) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{BASE_URL}/consent",
            json={"session_id": session_id, "consent": consent},
            timeout=10
        )
        response.raise_for_status()
        logging.debug(f"Consent response: {response.text}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Consent request failed: {str(e)}")
        return {"error": str(e), "session_id": session_id}

def send_message(message: str, session_id: str) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": message, "session_id": session_id},
            timeout=10
        )
        response.raise_for_status()
        logging.debug(f"Chat response: {response.text}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Chat request failed: {str(e)}")
        return {"error": str(e), "session_id": session_id}

def send_feedback(session_id: str, satisfaction: int, comments: str) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{BASE_URL}/feedback",
            json={"session_id": session_id, "satisfaction": satisfaction, "comments": comments},
            timeout=10
        )
        response.raise_for_status()
        logging.debug(f"Feedback response: {response.text}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Feedback request failed: {str(e)}")
        return {"error": str(e), "session_id": session_id}

def run_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    session_id = ""
    start_time = time.time()
    
    # Set consent
    consent_response = send_consent(session_id, True)
    logging.debug(f"Raw consent response: {consent_response}")
    if "error" in consent_response:
        return {"test_case": test_case["name"], "error": consent_response["error"]}
    
    session_id = consent_response["session_id"]
    
    # Send messages
    responses = []
    for message in test_case["messages"]:
        response = send_message(message["input"], session_id)
        logging.debug(f"Raw chat response for '{message['input']}': {response}")
        if "error" in response:
            responses.append({
                "input": message["input"],
                "error": response["error"],
                "expected_intent": message["expected_intent"],
                "actual_intent": None,
                "is_correct": False
            })
        else:
            actual_intent = response.get("analysis", {}).get("intent", {}).get("label")
            is_correct = actual_intent == message["expected_intent"]
            if not is_correct:
                logging.warning(
                    f"Intent mismatch: input='{message['input']}', "
                    f"expected={message['expected_intent']}, actual={actual_intent}"
                )
            responses.append({
                "input": message["input"],
                "response": response,
                "expected_intent": message["expected_intent"],
                "actual_intent": actual_intent,
                "is_correct": is_correct
            })
    
    # Send feedback
    feedback_response = send_feedback(session_id, test_case["satisfaction"], test_case["comments"])
    logging.debug(f"Raw feedback response: {feedback_response}")
    
    session_duration = time.time() - start_time
    
    return {
        "test_case": test_case["name"],
        "responses": responses,
        "session_duration": session_duration,
        "feedback": feedback_response
    }

TEST_CASES = [
    {
        "name": "Scientific Inquiry",
        "messages": [
            {"input": "What is the amygdala?", "expected_intent": "seeking_information"},
            {"input": "Tell me about mindfulness", "expected_intent": "seeking_information"}
        ],
        "satisfaction": 4,
        "comments": "Good info on brain terms"
    },
    {
        "name": "Emotional Support",
        "messages": [
            {"input": "I’m feeling sad again", "expected_intent": "emotional_support"},
            {"input": "I’m still sad", "expected_intent": "emotional_support"}
        ],
        "satisfaction": 3,
        "comments": "Empathetic but could suggest more actions"
    },
    {
        "name": "Coping Strategies",
        "messages": [
            {"input": "How do I cope with anxiety?", "expected_intent": "coping_strategies"}
        ],
        "satisfaction": 5,
        "comments": "Helpful breathing technique"
    },
    {
        "name": "Mixed Emotions",
        "messages": [
            {"input": "I’m happy but also anxious", "expected_intent": "emotional_support"}
        ],
        "satisfaction": 4,
        "comments": "Recognized mixed emotions well"
    },
    {
        "name": "Data Privacy",
        "messages": [
            {"input": "My name is John", "expected_intent": "personal_story"}
        ],
        "satisfaction": 4,
        "comments": "Handled personal info securely"
    }
]

def main():
    results = []
    for test_case in TEST_CASES:
        result = run_test_case(test_case)
        results.append(result)
    
    # Calculate metrics
    total_tests = sum(len(r["responses"]) for r in results if "responses" in r)
    correct_intents = sum(
        sum(1 for resp in result["responses"] if resp.get("is_correct", False))
        for result in results if "responses" in result
    )
    accuracy = (correct_intents / total_tests * 100) if total_tests > 0 else 0
    feedback_results = [r for r in results if "feedback" in r and "satisfaction" in r["feedback"]]
    avg_satisfaction = (
        sum(r["feedback"].get("satisfaction", 0) for r in feedback_results) / len(feedback_results)
        if feedback_results else 0
    )
    avg_session_duration = sum(
        r["session_duration"] for r in results if "session_duration" in r
    ) / len([r for r in results if "session_duration" in r]) if any("session_duration" in r for r in results) else 0
    
    logging.info(f"Test Results: Accuracy={accuracy:.2f}%, Avg Satisfaction={avg_satisfaction:.2f}, Avg Session Duration={avg_session_duration:.2f}s")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()