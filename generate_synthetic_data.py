import json
from collections import Counter
from transformers import pipeline

def generate_synthetic_examples(prompt, label, num_examples=20):
    generator = pipeline("text-generation", model="gpt2", device=-1)
    outputs = generator(prompt, num_return_sequences=num_examples, max_length=50, truncation=True)
    examples = [{"text": output['generated_text'].strip(), "label": label} for output in outputs]
    return examples

def balance_dataset(input_file, output_file, target_per_label=100):
    with open(input_file, 'r') as f:
        dataset = json.load(f)

    label_counts = Counter(item['label'] for item in dataset)
    print("Initial label distribution:", label_counts)

    prompts = {
        0: "Generate a friendly greeting like 'Hi there!' or 'Hey, what's up?'",
        1: "Ask a question about mental health or neuroscience, like 'What is serotonin?'",
        2: "Express a negative emotion, like 'I'm feeling sad because' or 'I'm so anxious'",
        3: "Ask for ways to manage stress or anxiety, like 'How can I cope with stress?'",
        4: "Request mental health resources, like 'Where can I find a therapist?'",
        5: "Share a personal struggle, like 'My life has been tough because'",
        6: "Express a mental health crisis, like 'I feel like I can’t go on'",
        7: "Describe a physical symptom, like 'I have a headache' or 'My stomach hurts'"  # New intent
    }

    new_examples = []
    for label in range(8):  # Updated to include LABEL_7
        current_count = label_counts.get(label, 0)
        needed = target_per_label - current_count
        if needed > 0:
            print(f"Generating {needed} examples for label {label}...")
            batch_size = 20
            while needed > 0:
                num_to_generate = min(batch_size, needed)
                synthetic = generate_synthetic_examples(prompts[label], label, num_to_generate)
                new_examples.extend(synthetic)
                needed -= num_to_generate

    dataset.extend(new_examples)
    unique_dataset = []
    seen_texts = set()
    for item in dataset:
        if item['text'] not in seen_texts:
            unique_dataset.append(item)
            seen_texts.add(item['text'])

    final_dataset = []
    label_counts = Counter()
    for item in unique_dataset:
        label = item['label']
        if label_counts[label] < target_per_label:
            final_dataset.append(item)
            label_counts[label] += 1

    with open(output_file, 'w') as f:
        json.dump(final_dataset, f, indent=4)

    final_counts = Counter(item['label'] for item in final_dataset)
    print("Final label distribution:", final_counts)
    print(f"Total examples: {len(final_dataset)}")

if __name__ == "__main__":
    balance_dataset(
        input_file='intent_dataset.json',
        output_file='intent_dataset.json',
        target_per_label=100
    )