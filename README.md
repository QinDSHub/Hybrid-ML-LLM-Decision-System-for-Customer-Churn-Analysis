# Hybrid ML + LLM Retrieval System for Customer Churn Decisioning

A production-oriented churn prediction system that combines **structured machine learning**, **semantic text embeddings**, and a **retrieval-based KNN decisioning strategy**.  
This project began as a research-driven modeling workflow and was later refactored into a more deployable engineering system with FastAPI serving, Docker support, and Azure-oriented CI/CD scaffolding.

---

## Why this project matters

Customer churn prediction is often treated as a pure classification task. In practice, however, real business scenarios require more than raw prediction accuracy:

- data quality issues must be handled carefully
- structured and unstructured signals need to be combined meaningfully
- predictions should be interpretable enough to support decision-making
- the solution should be deployable, testable, and extensible

This project was designed with those practical constraints in mind.

---

## Project highlights

- Built a **hybrid prediction framework** combining numerical features and semantic text embeddings
- Designed a **retrieval-based KNN decisioning approach** instead of relying only on a traditional classifier
- Achieved **AUC = 0.936** on the validation dataset
- Refactored research-style code into a more **modular, production-oriented architecture**
- Exposed the model through a **FastAPI inference service**
- Added **Docker containerization** and **GitHub Actions deployment scaffolding for Azure**
- Introduced a lightweight **hash embedding mode** for smoke tests, demos, and CI-friendly workflows

---

## My contribution

The **core business logic**, **system core frameworks**, **data preprocessing**, **feature engineering**, and **model-related implementation** were independently developed by me.

To improve engineering readiness, parts of the system — including **FastAPI service wrapping**, **Docker containerization**, **modular refactoring**, and portions of the **deployment / MLOps scaffolding** — were completed with ChatGPT-assisted support, then reviewed, adapted, validated, and tested by me before integration into the project.

---

## Business and technical objective

The goal of this project is to predict customer churn in an automotive service context by learning from both:

- **structured behavioral and service features**
- **text-based semantic signals**

Rather than using a purely black-box classifier, the system retrieves similar historical users and makes predictions through a **nearest-neighbor decisioning process**, improving both practical interpretability and traceability.

---

## Modeling approach

### Data cleaning and filtering

The pipeline applies several business-driven cleaning steps, including:
- further standardization of `repair_type`
- not add internal vehicles into datasets to reduce bias
- filtering out non-active service visits such as:
  - warranty / PDI claims
  - accident repairs
  - mandatory maintenance
  - warranty-related services
- some invalid records filtering directly instead of filling up with statistics method
- imputation of missing or abnormal values using user-level median daily metrics
- users who had not actively returned to the service center for **three years** were labeled as churned and not put into model datasets for training or validation

### Feature engineering
The model combines both **numerical** and **textual** signals.

#### Numerical features
Different preprocessing strategies were applied depending on feature distribution:

- `RobustScaler` for columns with extreme outliers
- `PowerTransformer` for highly skewed features
- `StandardScaler` for remaining columns

#### Text features
Relevant textual attributes were converted into semantic vectors using an **OpenAI text embedding model**.

### Feature fusion
Numerical features and text embeddings were concatenated with weighting:

- Numerical features: **70%**
- Text features: **30%**

L2 normalization was then applied to ensure consistent vector scaling.

### Prediction strategy
Instead of training a standard classifier, the system performs:

1. cosine similarity retrieval of Top-k nearest users
2. KNN-style majority voting
3. final churn prediction based on neighbor consensus

This makes the prediction logic easier to inspect and explain in business terms.

---

## Results

On the validation dataset, the hybrid retrieval-based approach achieved:

- **AUC**: 0.936
- **Precision**: 0.9256
- **Recall**: 0.9232
- **F1-score**: 0.9244
- **Accuracy**: 0.9383

A key takeaway from this work is that **high-quality data cleaning, careful feature design, and a well-structured retrieval mechanism** can outperform unnecessarily complex modeling stacks in real-world business problems.

---

## Additional experiments

I also explored several alternative modeling directions to better understand the trade-offs between accuracy, efficiency, and deployment practicality.

- Replacing OpenAI embeddings with an offline `sentence-transformers` model reduced external dependency, but performance dropped to an AUC at **0.90**
- Applying **PCA** to OpenAI text embeddings improved compactness, but significantly hurt performance, reducing AUC to **0.81**
- I also ran text feature ablation experiments to evaluate feature importance and reduce token cost. This allowed me to retain an AUC of **0.935**, only **0.001 below the best model**, while improving runtime efficiency and lowering inference cost

These experiments reinforced an important conclusion: for this use case, semantic text information adds real predictive value, but careful feature selection is essential for balancing performance and cost.

However, from a modeling perspective, another promising direction is to better integrate **LLM reasoning** with **traditional ML prediction**, so that the system can simultaneously deliver:

- strong predictive accuracy  
- persuasive, human-readable explanations  
- practical marketing or customer-retention recommendations  

I also explored several versions of this idea based on version 1 and 2. For example, I experimented with optimizing **binned numerical features** before feeding them into an LLM-style reasoning pipeline, based on testing with an offline **sentence-transformer** setup. I also tested a **sliding-window labeling strategy** to see whether a more dynamic target design could improve alignment between reasoning outputs and observed customer behavior.

While the explanation quality and business recommendations were often strong and practically meaningful, the reasoning outputs still showed a noticeable gap from the predefined **rule-based labels** and **sliding-window labels** across multiple evaluation metrics. Although the results were not yet strong enough to publish, I believe this remains a highly valuable direction for future exploration—especially for building systems that are not only accurate, but also interpretable and actionable in real business settings.

---

## Engineering work in this Version

Version 3 focused on turning the original modeling workflow into a more engineering-ready system.

### Main improvements
- refactored script-based code into clearer modules
- separated training, inference, configuration, and deployment logic
- replaced heavier persistence dependency with a lighter `joblib` model bundle approach
- added FastAPI service endpoints for online prediction
- added Docker-based packaging for deployment consistency
- added CI/CD scaffolding for Azure Container Apps
- introduced `hash` embedding mode for lightweight testing and demos

> Note: the actual `model_bundle.joblib` file is not included in this repository because of its size.

---

## Project structure

```text
churn-prediction-fastapi/
├── app/
│   ├── api/main.py                    # FastAPI entrypoint
│   ├── core/config.py                 # configuration management
│   ├── schemas/predict.py             # request / response schemas
│   └── services/
│       ├── embeddings.py              # OpenAI / Hash embeddings
│       ├── feature_engineering.py     # preprocessing + feature construction
│       └── model.py                   # training, save/load, prediction
├── scripts/
│   ├── train_pipeline.py              # train model from raw CSV
│   ├── bootstrap_demo_model.py        # generate demo model bundle
│   └── run_api.py                     # local API startup
├── tests/
│   └── test_smoke.py                  # smoke tests
├── .github/workflows/
│   ├── ci.yml                         # CI pipeline
│   └── deploy-azure-containerapps.yml # Azure deployment workflow
├── azure/create_infra.sh              # Azure infra bootstrap
├── examples/sample_predict_raw.json   # sample API request
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Project Status

The FastAPI application has been fully implemented and successfully validated in a local environment. The Docker build and runtime workflow was also tested successfully. However, deployment through Azure ACR could not be completed because my school-associated Azure account does not have the necessary permissions. Nevertheless, the deployment steps are provided below and may be useful for users who have access to Azure ACR and GitHub LFS.

---

## Deployment Guide
### Local Setup and Testing
It is recommended to create a dedicated virtual environment for this project.

- Install dependencies: pip install -r requirements.txt

- Create your own .env file based on .env.example, and configure the following:
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=
OPENAI_MODEL=text-embedding-3-small
MODEL_BUNDLE_PATH=artifacts/model/model_bundle.joblib

- Run the training pipeline with real data:
python scripts/train_pipeline.py \
  --raw-dir data/raw \
  --artifact-dir artifacts/model \
  --processed-path data/processed/cleaned_data.csv \
  --embedding-provider openai

- Start the API service locally:
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

- Access the service:
Swagger UI: http://127.0.0.1:8000/docs
Health Check: http://127.0.0.1:8000/health

### Docker Deployment
- Launch Docker Desktop (Windows) and ensure you are logged in.

- Open Windows PowerShell and navigate to the project directory: cd D:\the_project\churn_model

- Build the Docker image locally: docker build -t churn-prediction-api:local .

- Set environment variables:
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="your_base_url"

- Run the container locally:
docker run --rm -p 8000:8000 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -e OPENAI_BASE_URL=$env:OPENAI_BASE_URL `
  -e MODEL_BUNDLE_PATH=/app/artifacts/model/model_bundle.joblib `
  churn-prediction-api:local

Note: Ensure that artifacts/model/model_bundle.joblib exists before deployment. This file is not included in the repository due to its size.

- After startup, access the service:
http://localhost:8000
Swagger UI: http://localhost:8000/docs

### Azure Deployment (Public & Private Options)
In general, deploying with Azure Container Registry (ACR) requires appropriate permissions or a paid service plan.

For public deployment, you can push your Docker image to a public registry (e.g., Docker Hub) and deploy from there. Below are the steps:

- Push Image to Docker Hub
Ensure the Docker workflow in Section 2 runs successfully.

- In Windows powershell, log in to Docker Hub: docker login

- Tag the image: docker tag churn-prediction-api:local yourdockerhubname/churn-prediction-api:v1

- Push the image: docker push yourdockerhubname/churn-prediction-api:latest
- Verify the image in Docker Hub.
- Then, deploy via Azure CLI
- Ensure Azure CLI is installed before proceeding, and then in windows powershell to log in to Azure:az login
- Create a Resource Group: az group create --name churn-model --location uksouth
Note: uksouth is recommended due to broader service availability, but availability may depend on your account permissions.

- Create a Container Apps Environment:
az containerapp env create `
  --name churn-api-env `
  --resource-group churn-model `
  --location uksouth

- Create the Container App:
az containerapp create `
  --name churn-prediction-api `
  --resource-group churn-model `
  --environment churn-api-env `
  --image docker.io/yourdockerhubname/churn-prediction-api:latest `
  --target-port 8000 `
  --ingress external `
  --env-vars OPENAI_API_KEY=your_api_key OPENAI_BASE_URL=your_base_url MODEL_BUNDLE_PATH=/app/artifacts/model/model_bundle.joblib

- Azure Container Apps CLI allows you to create and configure applications with external ingress enabled via --ingress external, while specifying the target port.

- Retrieve the application URL:
az containerapp show `
  --name churn-prediction-api `
  --resource-group churn-model `
  --query properties.configuration.ingress.fqdn
  
- Access the deployed service:
https://<your-container-app-domain>

---

## Future directions
One of my next priorities is to continue modularizing the core pipeline and integrate it into Agentic Skill Workflows for automated orchestration. Potential examples include:<br>
•	triggering incremental prediction when new data arrives<br> 
•	running scheduled full retraining and model refresh<br> 
•	automating evaluation, validation, and deployment as reusable workflows<br> 
I am also interested in comparing this approach with Multi-Agent Workflows to better understand which architecture is more effective for maintainability, flexibility, and operational efficiency in production ML systems.

---

## Tech stack
•	Python<br>
•	Pandas / NumPy <br>
•	scikit-learn <br>
•	FastAPI <br>
•	Uvicorn <br>
•	GitHub Actions <br>
•	OpenAI Embeddings<br>
•	LLM infrastructure<br>
The FastAPI application has been implemented and locally validated. Docker packaging and Azure deployment scaffolding have also been prepared as part of the engineering refactor, but end-to-end validation and cloud deployment have not yet been fully completed.

---

## Closing note
This project reflects a principle I strongly value:
In many real-world business problems, strong data cleaning, thoughtful feature engineering, and a simple but well-designed retrieval strategy can outperform unnecessary complexity.
It also reflects my interest in designing practical intelligent systems that combine my expertise in ML/DL, LLMs, and Agentic AI.

---

## Discussion
Feedback, technical suggestions, and collaboration are always welcome.
