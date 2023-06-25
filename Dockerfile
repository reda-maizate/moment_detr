FROM continuumio/miniconda3

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "moment_env", "/bin/bash", "-c"]

RUN apt-get update && apt-get install git
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

ENV AWS_REGION=${AWS_REGION}
ENV SQS_QUEUE_NAME=${SQS_QUEUE_NAME}
ENV REDIS_HOST=${REDIS_HOST}
ENV REDIS_PORT=${REDIS_PORT}
ENV REDIS_USERNAME=${REDIS_USERNAME}
ENV REDIS_PASSWORD=${REDIS_PASSWORD}

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "moment_env", "python", "moment_detr/run_on_video/run_on_aws.py"]