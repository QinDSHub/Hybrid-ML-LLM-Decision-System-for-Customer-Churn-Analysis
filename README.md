# Main Optimizations of Version 2 for the Customer Churn Analysis<br>
This version focuses on data quality improvements, feature engineering, and a hybrid retrieval-based prediction strategy.<br>

## Feature Cleaning: repair_type<br>
The repair_type feature was further standardized and cleaned. It is now used for:<br>
•	Data filtering<br>
•	Text-based semantic representation<br>
________________________________________
## Removal of Internal Vehicles<br>
Internal vehicles were removed from the dataset to prevent potential bias in the analysis and modeling process.
________________________________________
## Filtering Non-Active Service Visits<br>
Only active service visits were retained. Records related to passive visits were removed.<br>
Examples of passive visits include:<br>
•	Warranty/PDI claims<br>
•	Accident repairs<br>
•	Mandatory maintenance<br>
•	Warranty-related services
________________________________________
## Additional Data Cleaning<br>
Further cleaning was applied to noisy or problematic records:<br>
•	day_diff: Invalid records were removed<br>
•	mile_diff: Missing or abnormal values were imputed using the user's median daily mileage (day_speed_median)
________________________________________
## Churn Labeling Strategy<br>
Users who have not actively visited the service center for three years were labeled as: churn = 100%<br>
This provides a clearer signal for churn identification and improving model calculating efficiency.
________________________________________
## Feature Engineering<br>
Both numerical features and text features were incorporated into the model.<br>
Numerical Features<br>
•	Standardized using scaling<br>
•	The scaler was fit on the entire dataset to ensure consistent scaling across all samples<br>
Text Features<br>
•	Converted into 1536-dimensional embeddings using the OpenAI text embedding model<br>
•	These embeddings capture semantic similarity
________________________________________
## Feature Fusion<br>
Numerical and text embeddings were concatenated horizontally.<br>
Feature weights were applied:<br>
Numerical features: 70%<br>
Text features: 30%<br>
After concatenation, L2 normalization was applied and made sure fit on full dataset to ensure consistent vector scaling, and after that to split into train and valid.
________________________________________
## Why LLM Inference Was Not Used<br>
In the previous version, I transformed numerical features into binned textual features and fed them into an LLM. However, a very important and fundamental concept has been overlooked: text embedding models struggled to distinguish values such as “1-year car age” vs “11-year car age”, actually “1-year car age” should be much more similar with ‘2-year car age’ rather than ’11-year car age’.<br>
While LLMs are extremely powerful for text reasoning, this task is primarily driven by structured numerical signals, so relying on LLM-based inference did not provide meaningful benefits. This led to the current hybrid approach.
________________________________________
## Training and Validation Split<br>
•	The dataset was split into training and validation sets, the size of total dataset is 55129 samples, and 10% as valid dataset.<br>
•	Custom Training embeddings were stored in ChromaDB for vector similarity search
________________________________________
## Prediction Method (KNN-Style Retrieval)<br>
Instead of using a traditional classifier, this version adopts a vector similarity retrieval strategy.<br>
For each sample in the validation set:<br>
1.	Retrieve the Top-10 most similar users using cosine similarity<br>
2.	Apply a KNN-style majority voting strategy<br>
3.	Assign the majority label as the final prediction, the threshold is set as 0.4 will get the best AUC with 0.936
________________________________________
## Result<br>
Using this hybrid retrieval-based approach, the model achieved: AUC = 0.936
________________________________________
## Key Insight<br>
Occam's Razor: simplicity over complexity.<br>
Strong data cleaning, thoughtful feature engineering, and simple retrieval-based methods can sometimes outperform overly complex models, depending on the application context.<br>
More sophisticated models are often better suited for:<br>
•	Large-scale datasets<br>
•	Highly complex business scenarios<br>
•	Situations requiring fine-grained trade-offs between precision and recall
________________________________________
## Project Status<br>
This project is currently still in the exploratory and development stage.<br>
The system has not yet been fully wrapped into a multi-agent or production-ready pipeline, so the full codebase is not published at this moment.<br>
The repository and implementation details will be shared once development is completed.
________________________________________
## Discussion<br>
Feedback, reviews, and discussions are very welcome.<br>
If you are working on similar problems involving hybrid features, vector retrieval, or churn prediction, feel free to share your thoughts or suggestions.

