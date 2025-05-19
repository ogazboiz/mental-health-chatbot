from transformers import pipeline
classifier = pipeline("text-classification", model="./intent_model", tokenizer="./intent_model")
test_inputs = [
    "What is the amygdala?",
    "I’m feeling sad again",
    "How do I cope with anxiety?",
    "My name is John",
    "Hi there!",
    "Where can I find a therapist?",
    "I’m in a crisis"
]
for input_text in test_inputs:
    result = classifier(input_text)
    print(f"Input: {input_text}, Label: {result[0]['label']}, Score: {result[0]['score']:.3f}")