FROM python:3.8.10-slim

RUN apt-get update && \
apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0
RUN apt-get clean

EXPOSE 5000/tcp

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip
RUN pip3 install --upgrade pip
RUN pip3 cache purge
RUN pip3 --no-cache-dir install -r requeriments.txt
ENTRYPOINT ["python3"]
CMD ["run.py"]