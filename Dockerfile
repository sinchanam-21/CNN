FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY src/ src/
COPY cifar_net.pth .
COPY app.py .

# Expose the port Gradio runs on
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
