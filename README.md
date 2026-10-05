# Agentic Personal Finance Tracker

An AI agent that reads a bank statement, categorises every transaction with a
trained machine-learning model, flags suspicious activity, and answers questions
about your spending in plain language — without ever letting the language model
do the maths.

Built during a Vodafone internship.

> **Ask it things like:** *"What did I spend the most on last month?"* ·
> *"Am I overspending anywhere?"* · *"How did July compare to June?"* ·
> *"Is there anything unusual on my account?"*

---

## Features

- **Upload any bank CSV** — only `date`, `description` and `amount` are needed.
  No categories, no labels.
- **Automatic categorisation** — a trained classifier assigns one of 8 spending
  categories from the raw transaction text alone (92% accuracy).
- **Fraud and anomaly detection** — an Isolation Forest flags unusual
  transactions and explains *why* each one looks suspicious.
- **Conversational answers** — a chatbot built on Google Gemini answers
  budget questions in natural language.
- **Numbers you can trust** — every figure comes from pandas, not the LLM.
  Each answer has a *"Show the working"* panel with the tool that ran, its
  arguments and the exact table it read.
- **Works offline** — with no API key, or when the free quota runs out, the
  agent falls back to keyword routing and template answers. Every analysis
  still works.

---

## How the agent works

The design rule is simple: **the language model never calculates anything.**
Each question goes through three separate steps:

```
  Question ──► 1. ROUTE ──► 2. EXECUTE ──► 3. PHRASE ──► Answer
               (Gemini)     (pandas)       (Gemini)
               picks a      computes the   writes prose
               tool         real numbers   about the results
```

1. **Route** — the model sees the question and the four tool descriptions and
   returns JSON naming one tool. It sees no transaction data at this stage.
2. **Execute** — the chosen pandas function runs. This is ordinary code, so the
   numbers are exact.
3. **Phrase** — the model receives the question plus the computed results and
   writes the reply. It is instructed not to do any arithmetic of its own.

`python -m src.check_grounding` proves this holds: it extracts every money
figure from every answer and checks that each one appears in the tool output.

Every step fails safely. A missing API key, an exhausted quota, malformed JSON
or a hallucinated tool name each fall back to keyword routing and a template
answer built straight from the data. The LLM layer also walks a chain of
Gemini models, moving to the next one whenever a model is rate-limited.

### The four tools

| Tool | Answers |
|---|---|
| `spending_summary` | What was spent, on what, plus subscriptions and fixed bills |
| `find_overspending` | Which categories are above this account's own recent norm |
| `compare_months` | What changed between two months |
| `detect_suspicious` | Unusual transactions, each with plain-language reasons |

---

## Tech stack

| Area | Tools |
|---|---|
| Language | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn (Logistic Regression, Random Forest, Isolation Forest) |
| LLM | Google Gemini via `google-genai` (Google AI Studio free tier) |
| Interface | Streamlit |
| Synthetic data | Faker |
| Charts | Matplotlib |
| Config | python-dotenv |

---

## Results

### Spending categorisation

Trained on 1,240 synthetic Egyptian transactions across 8 categories: Food,
Groceries, Transport, Shopping, Bills & Utilities, Entertainment,
Health & Fitness, and Travel.

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Logistic Regression** | **92.0%** | **0.87** |
| Random Forest | 90.3% | 0.85 |
| *Baseline (always guess the most common category)* | *31.5%* | — |

### Fraud detection — synthetic data

32 fraudulent transactions hidden among 1,240.

| Method | Precision | Recall | PR-AUC |
|---|---|---|---|
| **Isolation Forest** | 0.62 | **0.72** | **0.690** |
| Z-score baseline | 0.67 | 0.06 | 0.148 |

The Z-score baseline catches only 6% of fraud. It only asks whether an amount
is large, so it misses card-testing entirely (those charges are EGP 20) and
cannot see fraud hiding among legitimately expensive purchases.

### Fraud detection — real data

The [ULB credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
on Kaggle: 492 frauds among 284,807 transactions.

| Method | Precision@100 | PR-AUC | ROC-AUC |
|---|---|---|---|
| **Logistic Regression (supervised)** | **82.0%** | **0.768** | **0.980** |
| Isolation Forest (unsupervised) | 1.0% | 0.040 | 0.937 |
| Isolation Forest (fit on legitimate only) | 0.0% | 0.025 | 0.938 |

### Key finding

Isolation Forest did well on the synthetic data and poorly on the real data.
**The algorithm didn't change — the features did.** On synthetic data it was
given engineered domain signals: foreign country, transaction velocity, amount
z-score within category, and hour of day. The Kaggle dataset only offers
PCA-anonymised components, where fraud overlaps with normal spending instead of
sitting in a sparse region. Most of the anomaly-detection result came from
feature engineering, not from the algorithm.

---

## Getting started

### 1. Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/ali-elharras/Agentic-Finance-Tracker.git
cd Agentic-Finance-Tracker

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. Add a Gemini API key (optional)

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) and
create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-key-here
```

Without a key, the app still runs and answers every question using keyword
routing and template answers.

### 3. Generate the data and train the models

The data and trained models aren't stored in the repo, so build them once:

```bash
python -m src.generate_data          # -> data/processed/transactions.csv
python -m src.export_sample          # -> data/processed/sample_bank_export.csv
python -m src.train_categorizer      # -> models/categorizer.pkl
python -m src.train_fraud_detector   # -> models/fraud_detector.pkl
```

### 4. Run the app

```bash
streamlit run app/app.py
```

Then open http://localhost:8501. Upload a bank statement CSV, or click
**Explore with sample data**.

---

## Using your own bank statement

Upload a CSV with these columns. Header names are matched case-insensitively.

| Column | Required | Notes |
|---|---|---|
| `date` | Yes | Transaction date |
| `description` | Yes | The raw text from your bank, e.g. `FAWRY BILL PAY VODAFONE EGYPT` |
| `amount` | Yes | Negative amounts are treated as money out |
| `channel`, `city`, `country` | No | Used by fraud detection when present |

For the demo, upload `data/processed/sample_bank_export.csv`. It is a realistic
bank export with **no category column and no fraud labels** — only dates, raw
descriptions and amounts — so the model has to work everything out itself.

---

## Other scripts

```bash
python -m src.analytics              # spending analytics demo
python -m src.agent                  # runs the agent on demo questions
python -m src.check_grounding        # verifies no answer invents a number
python -m src.llm                    # checks your API key works
python -m src.validate_on_kaggle     # real-data evaluation (see below)
```

To run the Kaggle evaluation, download `creditcard.csv` from the
[dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and
place it in `data/raw/`.

---

## Project structure

```
├── app/
│   └── app.py                  # Streamlit chatbot interface
├── src/
│   ├── agent.py                # Route → execute → phrase agent and its 4 tools
│   ├── llm.py                  # Gemini client with model fallback chain
│   ├── analytics.py            # pandas spending analytics
│   ├── features.py             # Feature engineering and merchant normalisation
│   ├── merchants.py            # Egyptian merchant catalogue for synthetic data
│   ├── generate_data.py        # Synthetic transaction generator
│   ├── export_sample.py        # Builds the unlabelled demo bank export
│   ├── train_categorizer.py    # Trains and compares the category classifiers
│   ├── train_fraud_detector.py # Trains Isolation Forest vs Z-score baseline
│   ├── validate_on_kaggle.py   # Evaluation on the real ULB fraud dataset
│   └── check_grounding.py      # Checks every answer's numbers against tool output
├── data/
│   ├── raw/                    # Original data (not committed)
│   └── processed/              # Generated data (not committed)
├── models/                     # Trained models (not committed)
├── presentation.html           # Project presentation slides
├── requirements.txt
└── run_app.bat                 # Windows launcher
```

---

## Development plan

The project was built in seven sprints:

| Sprint | Deliverable |
|---|---|
| 0 | Environment and project skeleton |
| 1 | Synthetic transaction generator (Faker) |
| 2 | Spending categorisation model |
| 3 | Fraud and anomaly detection |
| 4 | pandas analytics engine |
| 5 | Agent with 4 tools |
| 6 | Streamlit chatbot and demo |
