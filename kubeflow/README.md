# End-to-End MLOps Pipeline — Customer Churn Prediction

## Overview
This project demonstrates a production-grade MLOps pipeline for predicting customer churn using a real-world dataset. It covers the complete lifecycle of a machine learning system — from data versioning and experimentation to deployment and automation.

The system is designed to be:
- Reproducible
- Scalable
- Automated
- Deployment-ready

## Problem Statement
Customer churn prediction helps businesses identify customers who are likely to leave. This enables proactive retention strategies and improves business outcomes.

## Architecture
```
Raw Data → DVC → Preprocessing → Training → MLflow Tracking
                                   ↓
                            Model Registry
                                   ↓
                           FastAPI (Serving)
                                   ↓
                        Docker → Kubernetes
                                   ↓
                         CronJob (Automation)
```

## Tech Stack

| Category              | Tools                  |
|-----------------------|------------------------|
| Language              | Python                |
| Data Versioning       | DVC                   |
| Experiment Tracking   | MLflow                |
| API                   | FastAPI               |
| Containerization      | Docker                |
| Orchestration         | Kubernetes            |
| Pipeline Automation   | Kubernetes CronJob    |
| Pipeline Orchestration| Kubeflow Pipelines    |

## Project Structure
```
mlops-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   └── train.py
│
├── app/
│   └── main.py
│
├── kubeflow/
│   ├── pipeline.py
│   ├── compile.py
│   └── README.md
│
├── dvc.yaml
├── params.yaml
├── churn_pipeline.yaml
├── Dockerfile
├── k8s-deployment.yaml
├── k8s-service.yaml
├── k8s-cronjob.yaml
└── requirements.txt
```

## Workflow
1. **Data Versioning**
   - Dataset tracked using DVC
   - Ensures reproducibility and version control

2. **Data Pipeline**
   - Preprocessing (cleaning, encoding, scaling)
   - Train/test split

3. **Model Training**
   - Random Forest model
   - Metrics tracked: Accuracy, Precision, Recall

4. **Experiment Tracking**
   - MLflow logs: Parameters, Metrics, Model artifacts

5. **Model Serving**
   - FastAPI exposes `/predict` endpoint
   - Swagger UI available at `/docs`

6. **Containerization**
   - Docker image created for portability

7. **Deployment**
   - Deployed on Kubernetes using: Deployment, Service

8. **Automation**
   - Kubernetes CronJob triggers pipeline periodically
   - Enables automatic retraining

## Getting Started

1. **Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd mlops-churn-prediction
   ```

2. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run Pipeline (Manual)**
   ```bash
   dvc repro
   ```

4. **Start MLflow UI**
   ```bash
   mlflow ui
   ```
   Open: http://127.0.0.1:5000

5. **Run API**
   ```bash
   uvicorn app.main:app --reload
   ```
   Open:
   - API: http://127.0.0.1:8000
   - Docs: http://127.0.0.1:8000/docs

## Docker

**Build Image**
```bash
docker build -t churn-mlops .
```

**Run Container**
```bash
docker run -p 8000:8000 churn-mlops
```

## Kubernetes

**Deploy Application**
```bash
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
```

**Access Service**
http://localhost:30007

## Automation (CronJob)

**Apply CronJob**
```bash
kubectl apply -f k8s-cronjob.yaml
```

**Verify**
```bash
kubectl get cronjobs
kubectl get jobs
```

## Kubeflow Pipelines

Kubeflow Pipelines provides an orchestration layer for ML workflows. This project includes a simple pipeline that automates the preprocessing and training steps.

### Pipeline Components
- **preprocess_op**: Runs data preprocessing script (`src/data/process.py`)
- **train_op**: Runs model training script (`src/models/train.py`)

### Compiling the Pipeline
To compile the pipeline into a YAML file:
```bash
cd kubeflow
python compile.py
```
This generates `churn_pipeline.yaml` in the root directory.

### Running the Pipeline
Upload `churn_pipeline.yaml` to your Kubeflow Pipelines UI and run it, or use the Kubeflow CLI:
```bash
kfp pipeline upload -p churn-pipeline churn_pipeline.yaml
kfp run create -e <experiment-name> -p churn-pipeline
```

## Pipeline Automation Logic
Schedule Trigger → Run Preprocessing → Train Model → Log to MLflow → Register Model

## MLflow Tracking
- Tracks experiments and metrics
- Model versioning via registry
- UI can be run locally

## Future Improvements
- CI/CD integration (GitHub Actions)
- MLflow deployment as service
- Monitoring (Prometheus + Grafana)
- Data drift detection
- Feature store integration

## Key Learnings
- Building reproducible ML pipelines
- Managing experiments and models
- Deploying ML systems on Kubernetes
- Automating retraining workflows
- Understanding separation of: Training, Serving, Orchestration

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
This project is open-source and available under the MIT License.