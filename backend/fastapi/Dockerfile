# Use the official Python 3.11 image as the base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /fastapi

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy the current directory contents into the container at /fastapi
COPY . .

# Install the Python dependencies from requirements.txt
RUN pip install -r ./requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run your application
CMD ["python", "main.py"]
