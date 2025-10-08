#syntax=docker/dockerfile:1
FROM python:3.9-slim
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
RUN --mount=type=secret,id=azure_connection_string \
    dvc remote modify azurestorage connection_string "$(cat /run/secrets/azure_connection_string)" --local && \
    dvc pull -v
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]