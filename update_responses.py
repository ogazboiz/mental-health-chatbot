import json

with open("responses.json", "r") as f:
    responses = json.load(f)

responses["1"].extend([
    "I\u2019m gathering info on that topic. Can you share more details to narrow it down?",
    "Interesting question! Could you clarify what you\u2019re curious about?"
])

responses["5"].extend([
    "Thanks for sharing—it sounds meaningful. Want to tell me more about it?",
    "I appreciate you opening up. How can I help you with this?"
])

with open("responses.json", "w") as f:
    json.dump(responses, f, indent=4)

print("Updated responses for LABEL_1 and LABEL_5")