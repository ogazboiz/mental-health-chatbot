import json
from collections import defaultdict

# Mapping from Kaggle intents to your 7 labels
intent_mapping = {
    'greeting': 0, 'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0,
    'about': 1, 'skill': 1, 'creation': 1, 'learn-mental-health': 1, 'mental-health-fact': 1,
    'fact-1': 1, 'fact-2': 1, 'fact-3': 1, 'fact-5': 1, 'fact-6': 1, 'fact-7': 1, 'fact-8': 1,
    'fact-9': 1, 'fact-10': 1, 'fact-11': 1, 'fact-12': 1, 'fact-13': 1, 'fact-14': 1, 'fact-15': 1,
    'fact-16': 1, 'fact-17': 1, 'fact-18': 1, 'fact-19': 1, 'fact-20': 1, 'fact-21': 1, 'fact-22': 1,
    'fact-23': 1, 'fact-24': 1, 'fact-25': 1, 'fact-26': 1, 'fact-27': 1, 'fact-28': 1, 'fact-29': 1,
    'fact-30': 1, 'fact-31': 1, 'fact-32': 1,
    'sad': 2, 'stressed': 2, 'worthless': 2, 'depressed': 2, 'anxious': 2, 'scared': 2, 'death': 2, 'friends': 2, 'hate-me': 2,
    'problem': 3, 'no-approach': 3, 'learn-more': 3, 'user-agree': 3, 'meditation': 3, 'user-meditation': 3, 'pandora-useful': 3,
    'help': 4, 'fact-13': 4, 'fact-14': 4, 'fact-18': 4, 'fact-19': 4, 'fact-21': 4, 'fact-24': 4,
    'name': 5, 'default': 5,
    'suicide': 6
}

def generate_response_mapping(kaggle_file, output_file):
    # Load Kaggle dataset
    with open(kaggle_file, 'r', encoding='utf-8-sig') as f:
        kaggle_data = json.load(f)

    # Aggregate responses by label
    responses_by_label = defaultdict(list)
    for intent in kaggle_data['intents']:
        tag = intent['tag']
        if tag in intent_mapping:
            label = intent_mapping[tag]
            responses_by_label[label].extend(intent['responses'])

    # Remove duplicates and save
    response_mapping = {str(label): list(set(responses)) for label, responses in responses_by_label.items()}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(response_mapping, f, indent=4)

    # Print summary
    for label, responses in response_mapping.items():
        print(f"Label {label}: {len(responses)} responses")

if __name__ == "__main__":
    generate_response_mapping(
        kaggle_file='intents.json',
        output_file='responses.json'
    )