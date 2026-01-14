## Transformer-Based Text Summarization System

State-of-the-art abstractive summarization using Hugging Face Transformers.

## Features
- Model comparison: BART-large-cnn, T5-base, PEGASUS
- Fine-tuning on CNN/DailyMail
- Optuna hyperparam tuning + early stopping
- Advanced generation (beam search, length penalty)
- Metrics: ROUGE-1/2/L, BERTScore
- Deployment: FastAPI inference API

## Run
```bash
pip install -r requirements.txt
python text_summarization.py
