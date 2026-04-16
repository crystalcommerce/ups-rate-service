# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install dependencies without progress bars to avoid threading issues
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]