# Kazakh Profanity Detection in Reviews (DSSC5113 bonus project)

Binary classifier: given one Kazakh review/comment, predict **0 = no Kazakh profanity, 1 = profanity present**, and show the probability.
Model: character n-gram TF-IDF + class-weighted logistic regression, trained from scratch by me (scikit-learn). Final test: **P = 0.954, R = 0.982, F1 = 0.968** on 2,631 held-out comments (169 positives). See the label-audit caveat below: these numbers measure agreement with a lexicon-assisted labeling, not perfect ground truth.

> Author: Maiorov Artyom, student ID 25MD0189. Individual work.

## 1. Quick start (no retraining)

```bash
pip install -r requirements.txt
python predict.py "your Kazakh comment here"      # prints label, probability, threshold
python predict.py                                  # interactive: type comments, empty line to quit
```
or in Python:
```python
from src.kkprof import predict
predict("your comment")      # {'label': 0, 'prob': 0.0123, 'threshold': 0.355}
```
The saved model is `models/tfidf_lr.joblib` (TF-IDF vectorizer + classifier + threshold in one file)

## 2. Data

**Source.** Review text comes from [KazSAnDRA](https://huggingface.co/datasets/issai/kazsandra) (`issai/kazsandra`, `full.csv`, CC BY 4.0, gated: accept the conditions on Hugging Face and use your own token via `HF_TOKEN`). It has 180,064 Kazakh reviews from four domains (appstore, market, mapping, bookstore); about 176.4k remain after exact deduplication on normalized text.

**Relation to product reviews.** The *market* domain consists of product reviews. The other domains are other consumer reviews. I used all of them because real profanity is very rare and almost all of it occurs in app/game reviews. **Market has essentially no real profanity** (see Limitations), so it serves only as negatives and hard negatives. Metrics are reported per domain.

**Labels.** KazSAnDRA has sentiment ratings only, which do not match this task (profanity also appears in 5-star reviews), so I labeled for profanity myself:
1. Candidate search with word lists (used only to *find* candidates): a filtered Kazakh list from [BibbyResearch/ProfanityBench](https://huggingface.co/datasets/BibbyResearch/ProfanityBench) (MIT, gated; filtered list in `data/lexicon_profanitybench_kk.csv`), plus my own tier lists and a Russian-mat regex (`notebooks/`, flagging cell). 2,076 comments were flagged.
2. I reviewed the flagged test rows manually (all 169 auto-proposed positives accepted; 143 mild-insult-only rows left unlabeled and excluded).
3. Unflagged comments are labeled 0 by assumption (`label_source = assumed_negative`); a random sample of 98 unflagged test rows found no positives.

`label_source` values: `manual`, `auto_lexicon` (flagged rows accepted as 1), `assumed_negative`; rows with `excluded_unlabeled` are not used in any split.

**Labeling rules.**
- **1** = the comment contains an obscene/vulgar word, or an abusive slur/insult aimed at a person, group or developers, in Kazakh or code-switched Russian. Russian profanity inside a Kazakh/code-switched comment counts as 1.
- Masked or respelled variants count if identifiable (e.g. digit substitutions, `*`-masking, Latin letters).
- A negative review alone is 0. Label by presence of a profane word, not by sentiment.
- **0**: descriptive/anatomical use (e.g. a diaper review), religious/medical terms, mild words used as exclamation/praise.
- Mild insults (tier B) are 1 only when clearly abusive; rows that only matched tier B and were not reviewed are **excluded** (limitation: no mild-insult hard cases in test).

**Examples** (profanity masked here; the dataset contains the unmasked text; reviewer names are not stored in the repo):

| text (masked/illustrative) | label | why |
|---|---|---|
| `қ***қбас, істемейді` | 1 | vulgar insult |
| `ешекбас бл*ть` | 1 | masked Russian profanity |

**Splits** (seed 42, 70/15/15, grouped so near-duplicates stay together: normalized key + TF-IDF cosine ≥ 0.85 on the labeled pool; bulk negatives use exact-normalized keys; stratified by source and proposed-flag):

| split | rows | positives | negatives |
|---|---|---|---|
| train | 12396 | 788 | 11608 |
| val | 2753 | 173 | 2580 |
| test | 2631 | 169 | 2462 |

(from the split × label table printed by the dataset-building cell). No synthetic examples were used. Class balance is strongly skewed toward 0, handled with `class_weight="balanced"`. Duplicate removal: exact duplicates after normalization were dropped; 0 normalized texts occur in more than one split.

`data/labels_public.csv` lists `custom_id`, domain, label, `label_source` and split, so the exact dataset can be rebuilt from KazSAnDRA. Review text is **not** committed.

## 3. Model

- **Input:** `prep()` lowercases (NFKC), replaces `ё`, removes links/emails, and collapses runs of 3+ identical characters to 2. Punctuation and emoji are kept (they fall at word boundaries for the n-gram analyzer).
- **Features:** TF-IDF over character n-grams (2–5, `char_wb`, `sublinear_tf`, `min_df=2`, max 300,000 features), fit on **train only**.
- **Classifier:** logistic regression, `class_weight="balanced"`, `random_state=42`; `C` and the decision threshold were chosen on the **validation set** (best F1 of class 1). Selected: C = 30, threshold = 0.355.
- **Why this model:** profanity in informal Kazakh is spelling-variable, inflected and code-switched. Character n-grams handle unseen words, suffixes and misspellings without a tokenizer or a Kazakh embedding model; the model is small, fast, trainable on a CPU and easy to explain.
- **Unknown tokens / long comments:** there is no vocabulary of whole words, so unseen words still produce known character n-grams. Comments of any length are used whole, with no truncation (see the long-text error below).
- Seed 42 everywhere. Training settings: `models/config.json`. Dependencies: `requirements.txt`.

**Tried and rejected (preprocessing v2 and v3).** 
Both were designed after looking at v1's test errors, to fix respelled and masked profanity (Котакбас with к instead of қ, қ0тақбас, бл*ть). The rule, fixed before running them: adopt a variant only if its validation F1 is at least v1's (0.969).

v2: fold Kazakh letters (қ→к, ғ→г, ә→а, ү/ұ→у, і→и, ң→н, ө→о, һ→х) and handle masking (*, 0, @ inside a word) in preprocessing, then retrain. Validation F1 0.958.
v3: keep the original spelling and handle masking, but train on both the original and the folded copy of every training comment (train only; validation and test are not augmented), with the C grid extended to 100. Validation F1 0.955.

## 4. Results (test set, evaluated once)

Profanity class = 1, threshold 0.355, n = 2,631:

| precision | recall | F1 | PR-AUC |
|---|---|---|---|
| 0.954 | 0.982 | 0.968 | 0.994 |

Confusion matrix (`results/confusion_matrix.png`):

|  | pred 0 | pred 1 |
|---|---|---|
| **true 0** | 2454 (TN) | 8 (FP) |
| **true 1** | 3 (FN) | 166 (TP) |

By domain: appstore 1,447 rows (169 positives; TP 166, FP 7, FN 3); bookstore 76 (FP 1); mapping 224 (no errors); market 884 (no errors, but 0 positives, so recall is undefined there).
Saved: `results/metrics.json`, `results/test_predictions.csv` (ids, labels, probabilities, no text).

**What errors mean.** A *false positive* is a clean review flagged as profane: with automatic removal it silences a customer. A *false negative* is a profane review that gets through: it stays visible to other users. For moderation I would use the model to *flag for human review* and favor recall (a missed profane review is worse than an extra review to check), which is why the F1-optimal threshold (0.355) is below 0.5. If comments were auto-removed, precision would matter more and the threshold should be raised.

## 5. Error analysis and label audit

Of 11 test errors (8 FP, 3 FN), I inspected all (`results/test_errors_local.csv`, not committed because it contains review text).

**Label audit.** Several "false positives" appear to be real profanity the word lists missed, so they are label errors, not model errors. I did **not** change test labels after seeing predictions. Examples (masked):
1. `қ*тақбас` written with `к` instead of `қ` (p = 0.51): a misspelling of a vulgar word, correctly flagged by the model, missed by the lexicon.
2. `ешекбас бл*ть` (p = 0.61): Russian mat masked with `*`; my regex needed a word boundary after `бля`.
3. A variant of `қақбас` with a changed vowel and mixed case (p ≈ 1.0).
Consequence: true precision is probably higher than 0.954, and the test set contains more than 169 true positives. The model generalizes beyond the word list to spelling variants.

**Genuine mistakes and hard cases:**
4. **FP: garbled legal-style text** (p = 0.47). Long text copied from a regulation with no profanity. Likely cause: character n-grams overlapping a profanity stem. Improvement: lexicon-aware features or a whole-word check to confirm char-n-gram hits.
5. **FN: long, spammy comment containing a vulgar word** (p = 0.25). The profane token is one among hundreds of n-grams, and L2-normalized TF-IDF dilutes it. Improvement: score windows/sentences and take the maximum.
6. **FN: `... бок бар` in a praise sentence** (p = 0.33). `бок` can be a Russian word ("side"), so this is an ambiguous case and the auto-label itself is questionable. Improvement: context-aware model (fine-tuned multilingual transformer) and manual relabeling of ambiguous tier-A words.

## 6. Limitations

- Labels are lexicon-assisted (weak labels). All test positives were found through the lexicon, so F1 largely measures agreement with the word list; test negatives are mostly assumed, not verified (see audit above).
- Almost all positives are app/game reviews. Market (product reviews) has no real positives found, so performance on product reviews is untested, though it produced 0 false positives on 884 comments.
- Mild-insult-only rows were excluded; the test set has no mild-insult hard cases.
- Near-duplicate grouping is approximate.
- Spelling variants and masked words are handled only through character n-grams; recall on unseen obfuscations is unknown.

## 7. Reproduce

1. `pip install -r requirements.txt`; accept the dataset conditions for `issai/kazsandra` and `BibbyResearch/ProfanityBench`; `export HF_TOKEN=...` (never commit it).
2. Run in order: (lexicon + flagging → label sheets → split → build dataset → train → evaluate). The labeled sheets contain review text and are not in the repo; `data/labels_public.csv` plus KazSAnDRA is enough to rebuild the dataset.
3. Evaluation writes to `results/`. The test set was evaluated once.

Works in a fresh Colab runtime with the same steps.

## 8. AI assistance and my contributions

I used Claude (Anthropic) for help with code, data-preparation pipeline design and error-analysis drafting. I understand and can explain all submitted code.
