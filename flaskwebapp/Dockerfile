
# Use an official Python runtime as a parent image
# If first time building, use python; else use ml_api so that cache can be utilized
#FROM python:3.6-slim
#FROM ml_api:base
FROM ml_api_with_nlp:base
    
# Copy the current directory contents into the container at /app. Different from mount
ADD . /app

# Set the working directory to /app
WORKDIR /app
RUN apt-get -y update && apt-get -y install tk-dev vim&& rm -r /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt (not local file)
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN python -m spacy download en

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]