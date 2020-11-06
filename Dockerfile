FROM tensorflow/tensorflow:2.3.1-gpu-jupyter
RUN apt-get update && \
  apt-get install -y -q --no-install-recommends --no-install-suggests ffmpeg libsm6 libxext6 && \
  apt-get autoremove -y && \
  rm -rf /var/lib/apt/lists/*
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt