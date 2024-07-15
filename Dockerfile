# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT 8080
ENV STREAMLIT_SERVER_ENABLE_CORS false

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8080

# Command to run the application
CMD ["streamlit", "run", "app.py"]
