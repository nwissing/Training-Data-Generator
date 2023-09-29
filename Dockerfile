FROM python:3.7-slim


WORKDIR /app

# Copy the Data
COPY app.py /app
COPY index.html /app
COPY sam_vit_l_0b3195.pth /app

# Install the Libraries
RUN pip install Flask flask_cors matplotlib opencv-python-headless torch segment_anything torchvision

# Opens the Port 5000
EXPOSE 5000

# Starts the app.py after the container has been launched
CMD ["python", "app.py"]
