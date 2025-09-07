# Dockerfile for AIdetectorML
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create working directory
WORKDIR /app
# Copy model files
COPY DNN_results/ DNN_results/
COPY Bert_results/ Bert_results/
# Install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app files
COPY . .

# Expose port 8080 (required by Cloud Run)
EXPOSE 8080

#for debugging purposes
RUN ls -la /app

# Run the app
CMD ["gunicorn", "-b", ":8080", "app:app"]