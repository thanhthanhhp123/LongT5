from datasets import load_dataset
from transformers import AutoTokenizer
dataset = load_dataset("cnn_dailymail", "3.0.0")
tokenizer = AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")

prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["highlights"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenized_data(dataset):
    tokenized_cnn = dataset.map(preprocess_function, batched=True)
    return tokenized_cnn

if __name__ == "__main__":
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    tokenized_cnn = tokenized_data(dataset)