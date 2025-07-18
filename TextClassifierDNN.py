# AI vs Human Text Classifier using TensorFlow (Binary Classification)
# Using a simple DNN model for text classification
# Made to determine between human and ai written text
# Will compare to see improvement between this and a BERT model
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
#to handle text data
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
#to split data into training and testing sets
from sklearn.model_selection import train_test_split 
#for early stopping:
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
#For saving the tokenizer and model
import pickle
#For handling numerical data
import numpy as np
#For handling CSV files
import csv
#For handling regular expressions and cleaning text data
import re
#For handling file paths
import string
#For handling system output redirection (optional, for debugging)
import sys
#For random number generation (optional, for reproducibility)
import random
 #redirect stdout to a file named log.txt for debugging purposes
#log_file = open('log.txt', 'w')
#sys.stdout = log_file
#sys.stderr = log_file

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#42 : 65% accuracy

# Clean data, remove extra spaces, make everything lowercase, remove punctuation, special characters, and HTML tags just in case

def clean_data(text):
    #everything should be lowercase:
    text = text.lower()
    # Remove extra spaces
    text = ' '.join(text.split())
    # Remove punctuation and special characters
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    # Remove HTML tags or any artifacts
    text = re.sub(r'<[^>]+>', '', text)

    return text

  

#Coverting text data to numerical format
def prepare_data(cleaned_texts, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>") #limit to 10,000 words
        # Fit the tokenizer on the cleaned texts
        tokenizer.fit_on_texts(cleaned_texts)
    # Convert texts to sequences and pad them
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    # Pad sequences to ensure uniform input size
    padded = pad_sequences(sequences, padding='post', maxlen=300) 
    #return np array of padded sequences and the tokenizer
    return np.array(padded), tokenizer



# Building a simple neural network model for binary classification
# using a sequential model with the sigmoid activation function
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid') #binary classification output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    return model



def train_model(model, x_train, y_train, epochs=10, batch_size=32):
    early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True) #stop if no improvement in accuracy for 2 training epochs
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop])



def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

#for predicting new text data
def predict_text(model, tokenizer, text):
    cleaned = clean_data(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, padding='post', maxlen=300)
    prediction = model.predict(padded)[0][0]
    label = "Human" if prediction > 0.5 else "AI"
    print(f"Prediction: {label} ({prediction:.2f})")


def main():

    # Open the CSV file
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

    #Clean and prepare data
    cleaned_texts = [clean_data(text) for text in texts]
    #break into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(cleaned_texts, labels, test_size=0.2, random_state=42)
    # Convert texts to numerical format
    x_train, tokenizer = prepare_data(x_train)
    x_test, _ = prepare_data(x_test, tokenizer)

    #convert the labels to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    # Build and train the model
    model = build_model((x_train.shape[1],))
    train_model(model, x_train, y_train)

    # Evaluate the model
    evaluate_model(model, x_test, y_test)
    #For debugging purposes 
    #log_file.close()
    #sys.stdout = sys.__stdout__
    #sys.stderr = sys.__stderr__
    #print("Logging complete.")

    # Predict on a new text, this was from my eagle scout accpetance speech, it should be classified as human
    sample_text = """
As grateful as I am today to receive this award, the title of “Eagle Scout” is one of the least important things I have gotten out of my scouting career. Rather it is the trials and tribulations I went through, and their permanent effect on my character that was the real important takeaway.
"""
    predict_text(model, tokenizer, sample_text)

    #save the model and tokenizer incase we want to use it later
    model.save("DNN_results/ai_vs_human_classifier.keras")
    with open("DNN_results/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()


