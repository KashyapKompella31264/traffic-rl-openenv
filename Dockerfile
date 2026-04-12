FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir pydantic flask openai

# Train the model during build so q_table.pkl is populated
RUN python train.py

EXPOSE 7860

CMD ["python", "-m", "server.app"]