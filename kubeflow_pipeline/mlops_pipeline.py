from kfp import dsl
from kfp.v2 import compiler  # ✅ Use v2 compiler

# -------------------------------
# Data Processing Component
# -------------------------------
@dsl.container_component
def data_processing_op():
    return dsl.ContainerSpec(
        image="sriram456/my-mlops-appp:latest",
        command=["python", "src/data_processing.py"],
    )


# -------------------------------
# Model Training Component
# -------------------------------
@dsl.container_component
def model_training_op():
    return dsl.ContainerSpec(
        image="sriram456/my-mlops-appp:latest",
        command=["python", "src/model_training.py"],
    )


# -------------------------------
# Pipeline Definition
# -------------------------------
@dsl.pipeline(
    name="MLOps Pipeline",
    description="Pipeline with data processing + model training"
)
def mlops_pipeline():

    data_task = data_processing_op()
    model_task = model_training_op()
    model_task.after(data_task)


# -------------------------------
# Compile Pipeline (KFP v2)
# -------------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path="mlops_pipeline.yaml"
    )
