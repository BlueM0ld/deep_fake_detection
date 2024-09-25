FROM python:3.10.12

WORKDIR /usr/src/app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

EXPOSE 3001

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"


CMD ["python", "./src/server.py"]