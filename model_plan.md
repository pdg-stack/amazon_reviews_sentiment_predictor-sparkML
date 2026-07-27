# Model Plan

This document explains **why** the sentiment-prediction pipeline is built the way it is — the reasoning behind each step, not just what the code does. It's written for someone new to NLP/ML pipelines, not just for someone reading the code.

## The problem
Each Amazon review comes with a `polarity` label: `1` (negative) or `2` (positive), plus free-text `title` and `text`. We're predicting sentiment (negative/positive) from the review — a binary text-classification problem.

## Why title + body are combined into one `review_text` field
`preprocess.py` concatenates `title` and `text` into a single `review_text` column (`concat_ws(" ", title, text)`, which gracefully drops the title when one isn't present) before anything else happens. The reasoning: a review's title is often a *concentrated* sentiment signal on its own — someone titling a review "Terrible, don't buy!!" has told you most of what you need to know before the body even starts. Throwing the title away would discard a cheap, high-signal feature; combining it with the body instead of modeling it as a separate feature keeps the rest of the pipeline (Tokenizer onward) unchanged — the model just sees more (and often more concentrated) text per review. Every stage below — and `train.py`, `predict.py`, and the API's `/predict` endpoint — operate on `review_text`, never on `text` alone, so training and serving always see the same kind of input.

## Why this pipeline, stage by stage

### 1. Tokenizer
Splits `review_text` into individual words ("This product is great" -> `["this", "product", "is", "great"]`). Machine learning models work with numbers, not sentences, so the first step in any text pipeline is breaking text into discrete units (tokens) that can eventually be turned into numeric features.

### 2. StopWordsRemover
Drops extremely common words ("the", "is", "a", ...) that carry little sentiment signal on their own. Removing them shrinks the vocabulary and lets the model spend its capacity on words that actually distinguish positive from negative reviews (e.g. "great", "terrible", "disappointed").

### 3. HashingTF (why hashing, not a plain word-count vocabulary)
Converts the remaining words into a fixed-size numeric vector by hashing each word to a bucket index. The alternative — building an explicit vocabulary (`CountVectorizer`) that maps every unique word to its own column — needs a full pass over the data to build that vocabulary and grows without bound as new words appear. Hashing trades a small, usually negligible risk of two different words landing in the same bucket ("hash collisions") for a fixed memory footprint and no separate vocabulary-building pass — a good trade at the scale of ~3.6M reviews.

### 4. IDF (Inverse Document Frequency)
Re-weights the hashed term-frequency vector so that words appearing in *almost every* review (even after stopword removal) count for less, and words that are rarer-but-informative count for more. TF alone treats "good" and "phenomenal" as equally important if they appear the same number of times; IDF recognizes that rarer words are often more discriminative.

### 5. LogisticRegression
The classifier itself. Chosen as the baseline over more complex models (e.g. deep neural nets) because: it's fast to train even on millions of sparse text-vector rows, it's directly interpretable (each feature gets a signed weight — you can inspect which hashed word-buckets push toward "positive" vs "negative"), and it's a strong, well-understood baseline for text classification. Once this baseline works end-to-end, swapping in a fancier classifier later is a one-line change (the rest of the pipeline stays the same).

## Why cross-validation + hyperparameter tuning
A single train/test split tells you how one specific model configuration performs on one specific slice of data — it doesn't tell you whether that configuration was actually the best choice, or whether the result was a bit of luck in how the data happened to split.

- **K-fold cross-validation (k=5)**: instead of one train/test split, the training data is split into 5 folds; the model trains on 4 and validates on the 1 left out, five times (rotating which fold is held out). The average performance across all 5 folds is a much more reliable estimate than any single split.
- **Hyperparameter grid** (`ParamGridBuilder`): rather than guessing values for `HashingTF.numFeatures`, `LogisticRegression.regParam`, and `elasticNetParam`, we try a small grid of combinations (2 x 2 x 2 = 8) and let cross-validation tell us which combination generalizes best — rather than which one merely fits the training data best.
- **`CrossValidator`** ties these together: it runs the 5-fold evaluation for *every* combination in the grid and returns the pipeline refit with the best-performing combination.

## Why a separate train/validation split *on top of* cross-validation
Cross-validation already gives an estimate of how well each hyperparameter combination generalizes — so why also carve out a validation set?

- The 80% "training" split is what `CrossValidator` folds internally.
- The 20% "validation" split is data the tuning process **never sees at all** — not even as a CV fold. After `CrossValidator` picks a winner, we check that winner once against this fresh slice as a sanity check that the CV process wasn't itself misleading (e.g. due to some subtle data leakage across folds).
- The fully separate `test.parquet` (the file `evaluate.py` uses) is **never touched during training or tuning at all** — it's reserved purely to report a final, unbiased estimate of how the finished model performs on data it has never influenced in any way. Three tiers — CV folds, validation split, held-out test set — each answering a slightly different question, and each protecting against a different way a model's reported performance can be misleadingly optimistic.

## Where MLflow fits in
Every `CrossValidator` run here means fitting 8 hyperparameter combinations x 5 folds = 40 individual models. Comparing 40 sets of metrics by hand doesn't scale — MLflow logs each run's parameters and metrics automatically, which is also the data source for the "did tuning actually help" chart in `evaluate.py` (baseline vs. best-tuned accuracy/AUC side by side).
