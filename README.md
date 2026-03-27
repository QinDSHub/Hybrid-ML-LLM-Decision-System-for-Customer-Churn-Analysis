# Implementation Overview

## Data Preprocessing

Data cleaning aligned with business were applied to ensure consistency and reliability for downstream modeling.

## Feature Engineering & Churn Label Definition

Churn is defined based on business logic:

max_date = estimated_last_visit_date + median_visit_interval + 3-month buffer

•	If a customer has not visited by max_date, they are labeled as churned (1), otherwise, they are labeled as active (0).

•	A binary classification setup is recommended. Although a three-class setup was explored, the LLM consistently converged to binary outputs. An optional “uncertain” class can be introduced via prompting when model confidence is low.

•	For feature engineering, a binning strategy aligned with business logic is applied to all features to normalize distributions and improve model stability and interpretability.

•	The attached validation output is provided for reference. The query section reflects the feature engineering results. All data has been anonymized, so it is safe to review.

## Top-K Feature Extraction (GBDTs)

LightGBM is employed for churn feature importance analysis, enabling downstream reasoning with a reduced and more informative feature set.

## Structured Text Transformation

Selected Top-K features are transformed into structured textual representations.

Both the raw data and generated text are in Chinese, reflecting the business context of this implementation.

## Vector Store Construction

Each customer profile is treated as an individual document and embedded using an embedding model.

Chroma is used as the vector store for similarity search.

## RAG-Based Retention Reasoning

For each customer profile in the validation set:

•	Top-N most similar customers are retrieved using cosine similarity

•	Retrieved profiles are provided to the LLM as RAG context for churn/retention reasoning based on customers in validation dataset.

## Evaluation & Iterative Optimization

•	Baseline performance with one hundred customers: AUC = 0.873, precision = 1, 	recall = 0.7463,	f1_score = 0.8547,	accurate = 0.83;

•	Manual review of misclassified samples revealed several optimization opportunities:

&nbsp;&nbsp;o	Automatically label customers with no visits for over three years as churned across train/validation/test sets, bypassing RAG and inference because RAG benefits from relevant volume, not sheer volume.

&nbsp;&nbsp;o	Strengthen preprocessing by removing accident-related repair records, which are non-habitual and introduce noise

&nbsp;&nbsp;o	Feed manually reviewed misclassified samples back into the RAG knowledge base for continuous improvement

&nbsp;&nbsp;o	Selecting stronger OpenAI embedding and chat models is expected to further improve performance. In this implementation, text-embedding-3-small and gpt-5-nano are used primarily for cost efficiency when operating at scale datasets.

&nbsp;&nbsp;o	Future exploration:

&nbsp;&nbsp;&nbsp;- Adopt a sliding-window–based churn labeling strategy instead of a purely time-statistics–based approach. Different business scenarios require different churn definitions, and the choice of labeling methodology has a direct and significant impact on model behavior, performance, and interpretability.

&nbsp;&nbsp;&nbsp;- Embed ML features directly as vectors instead of structured text, and hybrid of text and vector by weights.

&nbsp;&nbsp;&nbsp;- Explore deep learning approaches (e.g., Transformers) for embedding-based feature engineering. While deep learning can offer richer representations than classical ML, it often poses challenges for business-level explainability. This exploration is driven by curiosity and experimentation, with the goal of evaluating practical trade-offs rather than immediate production adoption.

## Agentic Workflow

A lightweight agentic pipeline is adopted:

•	The first agent performs reasoning, explanation, and recommendation via the LightGBM and LLM-Openai

•	Outputs are passed to downstream agents for further processing and integration

## Ongoing Work

I will continue to share:

•	Iteration details for each version

### •	Final, cleaned, and summarized code after completing the exploratory phase (code of release/v1.0 have been uploaded on 24th March 2026, run directly with automatic workflow by "python run.py")

•	An additional exploration is underway: building a handwriting OCR web app using RedNote.OCR, which has so far delivered the best performance in recognizing my handwritten text while preserving structural layout, enhanced further through LoRA fine-tuning.

•	The current AI landscape enables exciting experimentation that tightly integrates ML, DL, and agentic workflows.

## Data Compliance

Leveraged historical internal business data from Shanghai UnicData for this vehicle customer retention analysis, fully processed in-house and compliant with GDPR and data privacy requirements.

Happy to share, discuss and exchange ideas further 🚀
