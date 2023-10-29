import torch
from transformers import AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    DataCollatorForSeq2Seq, AutoTokenizer
import numpy as np
from datasets import load_dataset
import numpy as np
from model import load_model
from dataset import tokenized_data
# from utils import compute_metrics
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
dataset = load_dataset("cnn_dailymail", "3.0.0")
# model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
model = load_model()
tokenizer = AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["highlights"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_cnn = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="pszemraj/long-t5-tglobal-base-16384-book-summary")

training_args = Seq2SeqTrainingArguments(
    output_dir="mymodel",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_cnn["train"],
    eval_dataset=tokenized_cnn["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model()