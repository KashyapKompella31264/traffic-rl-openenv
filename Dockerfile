FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir pydantic flask

EXPOSE 7860

CMD ["python", "server.py"]