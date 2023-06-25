FROM continuumio/miniconda3

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "moment_env", "/bin/bash", "-c"]

RUN apt-get update && apt-get install git && apt-get install awscli -y
RUN git clone https://github.com/reda-maizate/moment_detr.git

#WORKDIR moment_detr/
#ENTRYPOINT ["tail", "-f", "/dev/null"]

ENV PYTHONPATH "${PYTHONPATH}:/moment_detr"

ARG SQS_QUEUE_NAME
ARG AWS_REGION
ARG REDIS_HOST
ARG REDIS_PORT
ARG REDIS_PASSWORD
ARG REDIS_USERNAME
ARG ACCESS_ID
ARG ACCESS_KEY
ARG AWS_QUEUE_OWNER_ID

ENV AWS_REGION=${AWS_REGION}
ENV SQS_QUEUE_NAME=${SQS_QUEUE_NAME}
ENV REDIS_HOST=${REDIS_HOST}
ENV REDIS_PORT=${REDIS_PORT}
ENV REDIS_USERNAME=${REDIS_USERNAME}
ENV REDIS_PASSWORD=${REDIS_PASSWORD}
ENV ACCESS_ID=${ACCESS_ID}
ENV ACCESS_KEY=${ACCESS_KEY}
ENV AWS_QUEUE_OWNER_ID=${AWS_QUEUE_OWNER_ID}

RUN aws configure set aws_access_key_id ${ACCESS_ID} && aws configure set aws_secret_access_key ${ACCESS_KEY} && aws configure set default.region ${AWS_REGION}

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "moment_env", "python", "moment_detr/run_on_video/run_on_aws.py"]