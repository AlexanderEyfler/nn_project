FROM continuumio/anaconda3:2024.06-1

WORKDIR /app

COPY . .

RUN conda env create -f environment.yml && \
    conda clean -afy

CMD ["conda", "run", "--no-capture-output", "-n", "nn_project_env", "python", "app.py"]
