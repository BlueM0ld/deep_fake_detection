FROM python:3.12

WORKDIR /streamlit

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN apt-get update && apt-get install -y ffmpeg

COPY . .

RUN pip install -r ./requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "–server.port=8501","--server.address=0.0.0.0"]

