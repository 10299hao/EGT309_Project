#run using "docker run team3-pipeline"
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["bash", "run.sh"]