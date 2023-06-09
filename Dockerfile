FROM python:3.10
RUN mkdir /fastapi
WORKDIR /fastapi
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD uvicorn app:app --host 0.0.0.0 --port 8000
# ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# docker rmi $(docker images -f "dangling=true" -q) --force  --> to remove untagged images

# docker run -p 8000:8000 -ti fastapi  --> run the fastapi image, on the localhost server, 
# with "t" which takes keyboard input into account so that "CTRL+C" works properly

# curl localhost:8000  --> display the message "alive" for example