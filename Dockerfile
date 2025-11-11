# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install required Python packages
RUN pip install -r package_versions.txt

# Expose the port your application runs on (e.g., Flask default is 5000)
EXPOSE 5000

# Command to run your application
CMD ["python", "app.py"]
