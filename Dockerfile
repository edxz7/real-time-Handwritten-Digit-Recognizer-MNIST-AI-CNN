FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt --upgrade

COPY app app/

EXPOSE 5042

CMD ["python", "app/server.py", "serve", "lenet-like"]