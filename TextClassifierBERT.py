#9/7/2025
#AI vs Human Text Classifier using TensorFlow (Binary Classification) 
#Using a Transformer-based BERT model for text classification
# Made to determine between human and AI written text
# Will compare to see improvement between this and a DNN model
#Make sure you are running this all on a GPU, torch wont say anything if you are not, but it will run much slower
#you can do print(torch.cuda.is_available()) to check if you have a GPU available
# Transformers
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Torch 
import torch

#for handling datasets
from datasets import Dataset

# Metrics (for eval metrics)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Data manipulation
import pandas as pd

# cleaming text data
import re

#handling CSV files
import csv

# Clean data, remove extra spaces, make everything lowercase, remove punctuation, special characters, and HTML tags just in case
# same as in DNN model

#update 9/7/2025: limiting cleaning to just potential html tags, as other factors can be relevant context for classification
def clean_data(text):
    #everything should be lowercase:
    #text = text.lower()
    # Remove extra spaces
    #text = ' '.join(text.split())
    # Remove punctuation and special characters
    #text = ''.join(char for char in text if char.isalnum() or char.isspace())
    # Remove HTML tags or any artifacts
    text = re.sub(r'<[^>]+>', '', text)

    return text

#convert into a hugginfface dataset
def prepare_dataset(texts, labels):
    # Create a pandas DataFrame
    df = pd.DataFrame({'text': texts, 'label': labels})
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    return dataset


#prepare the tokenizer and apply it to the dataset
def tokenizedata(data, tokenizer):
    return tokenizer(data['text'], padding="max_length", truncation=True, max_length=512) #increased to berts max length of 512, as much context as possible will hopefully increase accuracy


# Define evaluation metrics and return them
def computemetrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

#train the model
def train_model(tokenized_dataset, compute_metrics, tokenizer):
    #load the model, as transformers require a pre-trained model, due to training time and resources
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./Bert_results",
        evaluation_strategy="epoch",     
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,               
        warmup_ratio=0.1,              
        weight_decay=0.01,
        logging_steps=50,
        seed=42,
)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # Evaluate after training
    metrics = trainer.evaluate()
    print("Final evaluation metrics on the test set:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    # Save the model and tokenizer
    trainer.save_model("./Bert_results")
    tokenizer.save_pretrained("./Bert_results")
    with open("./Bert_results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

#for predicting new text data
def predict_text(model, tokenizer, text: str) -> str:
    #clean the text data
    cleaned = clean_data(text)
    tokens = tokenizer(cleaned, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        output = model(**tokens)
        logits = output.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    label = "Human" if prediction == 1 else "AI"
    print(f"Prediction: {label} (Logits: {logits.tolist()[0]})")
    return label

def main():


    #check if GPU is available(itll be slow either way tho)
    if torch.cuda.is_available():
        print("GPU is available, using it for training.")
    else:
        print("WARNING: GPU is not available, using CPU for training. This may be slow.")

    #Load dataset
    # Open the CSV file
    print("Loading dataset...")
    with open('./data/AI_Human.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if there is one

        texts = []
        labels = []

        for row in reader:
            # Assumes the first column is the full text, second is the label
            text = row[0].strip()
            label = int(float(row[1]))  # Converts '0.0'/'1.0' → 0/1, 0 = AI, 1 = Human
            texts.append(text)
            labels.append(label)


    # Example output
    print(f"Loaded {len(texts)} samples")
    print("First sample:") #print the first sample text and label just to make sure everything is working
    print("Text:", texts[0][:100], "...")
    print("Label:", labels[0])


    #clean up the text data
    cleaned_texts = [clean_data(text) for text in texts]

    # Prepare the dataset
    dataset = prepare_dataset(cleaned_texts, labels)

    # Tokenize the dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_dataset = dataset.map(lambda x: tokenizedata(x, tokenizer), batched=True, remove_columns=['text'])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels']) 
    # Train and save the model
    train_model(tokenized_dataset, computemetrics, tokenizer)
    print("success")
if __name__ == "__main__":
    main()


