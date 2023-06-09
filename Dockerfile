FROM python:3.10
RUN mkdir /fastapi
WORKDIR /fastapi
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD uvicorn app:app --host 0.0.0.0 --port 8000

# RUN mkdir /app/code
# COPY code/hello_world.py /app/code/hello_world.py
# WORKDIR /app
# CMD ["python", "code/hello_world.py"]