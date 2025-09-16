# --- 1. Base Image ---
# We use an official Python image. Using a specific version is good practice.
# The 'slim' variant is smaller and contains the minimum packages needed to run Python.
FROM python:3.11-slim

# --- 2. Set Working Directory ---
# This sets the default directory for all subsequent commands.
WORKDIR /app

# --- 3. Copy Requirements ---
# Copy the requirements file first to leverage Docker's layer caching.
# This way, dependencies are only re-installed if requirements.txt changes.
COPY requirements.txt .

# --- 4. Install System Dependencies ---
# We need to install libgl1 for OpenCV to work correctly.
RUN apt-get update && apt-get install -y libgl1

# --- 5. Install Dependencies ---
# We install the Python packages specified in the requirements file.
# --no-cache-dir reduces the image size.
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --no-deps torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# --- 6. Copy Application Code and Model ---
# Copy the rest of the application's code into the container.
COPY api.py .
# The model is a critical part of the application, so we copy it in.
COPY ./runs/final_model/weights/best.pt ./runs/final_model/weights/best.pt

# --- 7. Expose Port ---
# This informs Docker that the container listens on the specified network port at runtime.
# This must match the port uvicorn will run on.
EXPOSE 8000

# --- 8. Define Run Command ---
# This is the command that will be executed when the Docker container starts.
# We use 0.0.0.0 to make the server accessible from outside the container.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
