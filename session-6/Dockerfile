# set base image (host OS)
FROM python:3.8-slim

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt /tmp/

# install dependencies
RUN pip install -r /tmp/requirements.txt

# Copy code to the working directory
COPY ./src /app/src/
COPY ./data /app/data/

# command to run on container start
ENTRYPOINT ["python", "/app/src/entrypoint.py"]
# ENTRYPOINT ["tail", "-f", "/dev/null"]
