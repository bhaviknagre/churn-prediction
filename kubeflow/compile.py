from kfp import compiler          
from pipeline import churn_pipeline

compiler.Compiler().compile(      
    pipeline_func=churn_pipeline,
    package_path="churn_pipeline.yaml"
)
