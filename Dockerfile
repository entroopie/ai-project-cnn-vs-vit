FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

CMD ["python", "main.py"]
