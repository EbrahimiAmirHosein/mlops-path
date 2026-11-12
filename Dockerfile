FROM python:3.11-slim

WORKDIR /app


# TODO: copy requirements.txt FIRST so the install layer is cached
COPY requirements.txt .
# TODO: install dependencies with --no-cache-dir

RUN pip install --no-cache-dir -r requirements.txt

# TODO: copy the source code
COPY train.py .

# TODO: default command runs training and writes into /app/artifacts
CMD ["python", "train.py"]