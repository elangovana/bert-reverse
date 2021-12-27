# SageMaker PyTorch image
ARG device=gpu
ARG account_url=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training
ARG cuda=cu113
FROM ${account_url}:1.10.0-${device}-py38${cuda}-ubuntu20.04-sagemaker
ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY ./src  /opt/ml/code
RUN pip install -r /opt/ml/code/requirements.txt

