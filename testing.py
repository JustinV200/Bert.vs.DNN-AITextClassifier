import torch
from transformers import BertForSequenceClassification, BertTokenizer


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
    #reload and predict on the same text used in the DNN
    model = BertForSequenceClassification.from_pretrained("./Bert_results")
    tokenizer = BertTokenizer.from_pretrained("./Bert_results")

    sample_text = """
        As grateful as I am today to receive this award, the title of “Eagle Scout” is one of the least important things I have gotten out of my scouting career. Rather it is the trials and tribulations I went through, and their permanent effect on my character that was the real important takeaway.
    """

    predict_text(model, tokenizer, sample_text)

if __name__ == "__main__":
    main()