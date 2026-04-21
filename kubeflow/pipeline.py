from kfp import dsl
from kfp.dsl import component

@component
def preprocess_op():
    import subprocess
    subprocess.run(["python", "src/data/process.py"], check=True)

@component
def train_op():
    import subprocess
    subprocess.run(["python", "src/models/train.py"], check=True)

@dsl.pipeline(name="churn-pipeline")
def churn_pipeline():
    preprocess = preprocess_op()
    train = train_op()
    train.after(preprocess)

