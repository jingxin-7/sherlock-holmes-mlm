# Sherlock Holmes Masked Language Modeling

### AIAA 3102 — Artificial Intelligence | PyTorch & BERT-Tiny

A masked language modeling (MLM) project that predicts missing tokens in Sherlock Holmes passages using a pretrained BERT-Tiny encoder, a relative-position gap reader, and two complementary vocabulary prediction pathways.

The project explores parameter-efficient language modeling, transfer learning, vocabulary prediction, and cross-book generalization.

## Results

| Metric | Result |
|---|---|
| Final validation cross-entropy | **3.5033** |
| Initial clean-slate validation CE | 3.7787 |
| Improvement | **0.2754 CE** |
| Model parameters | 7,064,891 |
| Maximum permitted parameters | 9,000,000 |
| Vocabulary size | 30,522 tokens |
| Validation masked tokens | 4,208 |

*Results were obtained using the course-provided validation evaluator. Lower cross-entropy is better. Hidden-test performance is not reported.*

## Model Architecture

The final model, **Clean-Slate V2**, consists of four main components:

**1. Pretrained BERT-Tiny Encoder**

Uses `google/bert_uncased_L-2_H-128_A-2` to produce contextual token representations. Only permitted pretrained encoder weights are used.

**2. Relative-Position Gap Reader**

Constructs contextual representations for masked positions using surrounding token information and relative positions.

**3. Dual Prediction Pathways**

- **Path A — Frequency-Clustered Decoder:** Uses vocabulary clustering and contextual information initialized from authorized training text.
- **Path B — Embedding-Tied Decoder:** Uses a newly initialized Linear–GELU–LayerNorm prediction head whose vocabulary projection shares the encoder's pretrained input embeddings.

**4. Learned Probability Mixture**

A trainable gate combines the distributions from both prediction pathways to produce final vocabulary predictions.

The complete model contains approximately 7.06 million parameters.

## Training

The training procedure uses the three Sherlock Holmes texts supplied for the assignment:

- *The Adventures of Sherlock Holmes*
- *The Return of Sherlock Holmes*
- *Some later reminiscences of Sherlock Holmes*

All masked training examples are generated and stored in a static CSV before training. Inputs contain one to three masked tokens and use the BERT-Tiny vocabulary.

Training involves a base-model stage followed by training of the embedding-tied prediction pathway and probability mixture.

## Experiments and Findings

Several approaches were investigated, including:

- Decoder-focused training and data enrichment
- Relative-position contextual modeling
- Frequency-clustered vocabulary prediction
- Pretrained input/output embedding tying
- Transformer–BiGRU ensemble modeling
- Rare-token reweighting and frequency-aware corrections
- Confidence-aware mixture gating
- Teacher–student knowledge distillation

The largest improvements came from introducing the clean-slate contextual prediction architecture and the embedding-tied prediction pathway.

Held-out Sherlock Holmes experiments showed improvements in both common-token and rare-token prediction. However, predicting rarely supervised and never-supervised tokens remains challenging, and performance on substantially different unseen texts is uncertain.

## Repository Structure

```text
sherlock-holmes-mlm/
├── train_model.ipynb
├── student_model.py
├── v1_model.py
├── make_data.py
├── train.py
├── tied_train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment

The model was verified using an environment compatible with:

- Python 3.14.x
- PyTorch 2.13.x
- Hugging Face Transformers 5.20.x

Additional package specifications are listed in `requirements.txt`.

## Reproducibility and Data Availability

This GitHub repository is a **source-code showcase**, not a standalone executable reproduction package.

The trained model weights, generated training CSV, course-provided books, tokenizer resources, and permitted encoder-reference export are not distributed here.

Consequently, running the complete training notebook requires the original authorized course materials and corresponding resources. The validation results reported above were obtained from a separately verified, complete submission artifact.

## Limitations

- Performance may vary across authors, literary styles, and vocabulary distributions.
- Rare and previously unsupervised vocabulary remains difficult to predict.
- Held-out development results do not guarantee performance on the hidden evaluation set.
- The model is designed for masked-token prediction, not open-ended text generation.

## Acknowledgment

Developed as an individual artificial intelligence coursework project with AI-assisted coding and experimentation tools. Model development, testing, and validation followed the course's stated requirements and instructor clarifications.
