# Use a Python 3.12 slim-bookworm image as the base
FROM python:3.12-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy dependency management files
# pyproject.toml and uv.lock are copied outside src to ensure uv can find them
COPY pyproject.toml .
COPY uv.lock .

# Install uv if it's not already in the slim image and then install dependencies
# We assume 'uv' is used based on 'uv.lock' file presence.
# --system flag ensures installation into system site-packages, not a virtual environment
RUN pip install uv && uv sync --system

# Copy the src directory structure relevant to the application
# .dockerignore will ensure only necessary files are copied
COPY src ./src

# Expose the port FastAPI runs on
EXPOSE 8001

# Define the command to run the FastAPI application
# Using uvicorn directly. For production, gunicorn + uvicorn workers are often preferred.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]