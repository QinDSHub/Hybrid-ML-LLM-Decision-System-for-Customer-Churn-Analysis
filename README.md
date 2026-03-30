# Main Optimizations of Version 2 for the Customer Churn Analysis<br>
This version focuses on data quality improvements, feature engineering, and a hybrid retrieval-based prediction strategy.<br>

## Feature Cleaning: repair_type<br>
The repair_type feature was further standardized and cleaned. It is now used for:<br>
&emsp;•	Data filtering<br>
&emsp;•	Text-based semantic representation<br>
________________________________________
## Removal of Internal Vehicles<br>
Internal vehicles were removed from the dataset to prevent potential bias in the analysis and modeling process.
________________________________________
## Filtering Non-Active Service Visits<br>
Only active service visits were retained. Records related to passive visits were removed.<br>
Examples of passive visits include:<br>
&emsp;•	Warranty/PDI claims<br>
&emsp;•	Accident repairs<br>
&emsp;•	Mandatory maintenance<br>
&emsp;•	Warranty-related services
________________________________________
## Additional Data Cleaning<br>
Further cleaning was applied to noisy or problematic records, such as removing invalid records (e.g., negative day differences) and imputed missing or abnormal values (e.g., mileage) using each user’s median daily metrics.
________________________________________
## Churn Labeling Strategy<br>
Users who have not actively visited the service center for three years were labeled as: churn = 100%<br>
This provides a clearer signal for churn identification and improving model calculating efficiency.
________________________________________
## Feature Engineering<br>
Both numerical features and text features were incorporated into the model.<br>
Numerical Features<br>
&emsp;•	Standardized using scaling, the following preprocessing strategy was mainly used:<br>
&emsp;&emsp;- Outlier detection: RobustScaler applied to columns containing extreme values<br>
&emsp;&emsp;- Skewness detection: PowerTransformer utilized for highly skewed distributions<br>
&emsp;&emsp;- Default handling: StandardScaler used for all remaining columns<br>
&emsp;•	The scaler was fit on the entire dataset to ensure consistent scaling across all samples<br>

Text Features<br>
&emsp;•	Converted into high dimensional embeddings using the OpenAI text embedding model<br>
&emsp;•	These embeddings capture semantic similarity
________________________________________
## Feature Fusion<br>
Numerical and text embeddings were concatenated horizontally.<br>
Feature weights were applied:<br>
&emsp;•	Numerical features: 70%<br>
&emsp;•	Text features: 30%<br>
After concatenation, L2 normalization was applied and made sure fit on full dataset to ensure consistent vector scaling, and after that to split into train and valid.
________________________________________
## Why LLM Inference Was Not Used<br>
In the previous version, I transformed numerical features into binned textual features and fed them into an LLM. However, a very important and fundamental concept has been overlooked: text embedding models struggled to distinguish values such as “1-year car age” vs “11-year car age”, actually “1-year car age” should be much more similar with ‘2-year car age’ rather than ’11-year car age’.<br><br>
While LLMs are extremely powerful for text reasoning, this task is primarily driven by structured numerical signals, so relying on LLM-based inference did not provide meaningful benefits. This led to the current hybrid approach.
________________________________________
## Training and Validation Split<br>
•	Since the preprocessed dataset is unique to each vehicle user, we could safely perform a shuffling train-validation split to prevent label leakage.<br>
•	Custom Training embeddings were stored in ChromaDB for vector similarity search
________________________________________
## Prediction Method (KNN-Style Retrieval)<br>
Instead of using a traditional classifier, this version adopts a vector similarity retrieval strategy.<br>
For each sample in the validation set:<br>
&emsp;•	Retrieve the Top-k most similar users using cosine similarity<br>
&emsp;•	Apply a KNN-style majority voting strategy<br>
&emsp;•	Assign the majority label as the final prediction, the threshold is set as 0.4 will get the best AUC with 0.936
________________________________________
## Result<br>
Using this hybrid retrieval-based approach, the model achieved on total 10% valid datasets: AUC = 0.936，precision = 0.9256, recall = 0.9232, f1_score = 0.9244, accurate = 0.9383;

________________________________________
## Limitations and practical considerations<br>
While the approach performs well in many scenarios, several practical considerations are worth highlighting.<br>

First, the effectiveness of KNN-based similarity retrieval depends heavily on whether sufficiently similar users actually exist in the dataset. In certain edge cases, the nearest neighbors retrieved may still be weakly related to the target user. Directly transferring labels or behavioral signals from such neighbors may introduce noise rather than meaningful signal. A practical mitigation strategy is to introduce a similarity threshold, ensuring that only high-confidence neighbors contribute to downstream inference. In addition, multiple similarity or distance metrics can be incorporated to better monitor and validate the robustness of the retrieval process.<br>

Second, the choice of text embedding model should be aligned with the complexity and richness of the textual features. OpenAI embeddings generate Transformer-based representations with high dimensions, which provide strong semantic capacity but may be unnecessarily heavy for simple or limited textual inputs. In those cases, lighter embedding models with lower-dimensional representations may offer a more efficient trade-off between semantic fidelity and computational cost.<br>

Finally, an interesting extension of thinking is that the models developed can naturally serve as Modular Capability Providers (MCPs) within agent or multi-agent architectures, enabling them to be orchestrated as reusable components in larger decision or automation pipelines.
________________________________________
## Key Insight<br>
Following the principle of Occam’s Razor, simplicity often outperforms unnecessary complexity. In practice, high-quality data cleaning, thoughtful feature engineering, and well-designed retrieval mechanisms can sometimes deliver stronger results than increasingly complex modeling stacks.<br>
________________________________________
## Project Status<br>
This project is currently still in the exploratory and development stage.<br>
The system has not yet been fully wrapped into a multi-agent or production-ready pipeline, so the full codebase is not published at this moment.<br>
The repository and implementation details will be shared once development is completed.
________________________________________
## Discussion<br>
Feedback, reviews, and discussions are very welcome.<br>
If you are working on similar problems involving hybrid features, vector retrieval, or churn prediction, feel free to share your thoughts or suggestions.
