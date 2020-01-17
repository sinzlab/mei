FROM sinzlab/pytorch:v1.3.1-cuda10.1-dj0.12.4

WORKDIR /src/mei
COPY . .
RUN python3 setup.py develop