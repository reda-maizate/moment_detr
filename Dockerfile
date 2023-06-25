FROM continuumio/miniconda3

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "moment_env", "/bin/bash", "-c"]

RUN apt-get update && apt-get install git
#RUN git clone https://github.com/reda-maizate/moment_detr.git

#WORKDIR moment_detr/
#ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "moment_env", "python", "run_on_video/run_on_aws.py"]