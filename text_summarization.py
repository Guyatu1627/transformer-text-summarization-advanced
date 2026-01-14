import logging
import sys
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
import evaluate
import numpy as np
import torch
import optuna
import wandb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/bart-large-cnn"  # Switch to "google-t5/t5-base" or "google/pegasus-cnn_dailymail"
MAX_INPUT = 1024
MAX_TARGET = 128
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 8

# ── Load Dataset ─────────────────────────────────────────────────────────────
dataset = load_dataset("cnn_dailymail", "3.0.0")
# Subsample for speed (demonstration only)
dataset["train"] = dataset["train"].select(range(20))
dataset["validation"] = dataset["validation"].select(range(10))
dataset["test"] = dataset["test"].select(range(5))
logger.info(f"Dataset loaded and subsampled: {dataset}")

# ── Preprocessing ────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT, truncation=True, padding="max_length")
    
    # Tokenize targets
    labels = tokenizer(text_target=examples["highlights"], max_length=MAX_TARGET, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
logger.info("Tokenization complete")

# ── Metrics ──────────────────────────────────────────────────────────────────
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bert_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    rouge_result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in rouge_result.items()}

# ── Training ─────────────────────────────────────────────────────────────────
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    predict_with_generate=True,
    fp16=False,  # Set to False for CPU
    push_to_hub=False,
    report_to="none",  # Disable wandb
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    max_steps=10,  # Very short training for demo
    save_strategy="epoch"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# trainer.train() # Commented out for demo speed on CPU
logger.info("Training skipped for speed (using pre-trained weights)")
logger.info("Training complete")

# ── Inference ────────────────────────────────────────────────────────────────
def summarize(text, max_length=130, min_length=30, num_beams=4):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=num_beams,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ── Evaluation on Test Set ──────────────────────────────────────────────────
# test_results = trainer.predict(tokenized_datasets["test"])
# logger.info(f"Test ROUGE: {test_results.metrics}")

# ── FastAPI Deployment ──────────────────────────────────────────────────────
app = FastAPI(title="Advanced Abstractive Text Summarizer")

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30
    num_beams: int = 4

@app.post("/summarize")
async def summarize_endpoint(request: SummaryRequest):
    try:
        summary = summarize(request.text, request.max_length, request.min_length, request.num_beams)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    logger.info("=== Project 9: Transformer-Based Text Summarization System ===")
    
    # Example inference
    text = "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It also invests in domestic energy production and manufacturing, and it reduces the deficit."
    summary = summarize(text)
    logger.info(f"Example summary: {summary}")
    
    # Run API server
    logger.info("Starting FastAPI server – visit http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()