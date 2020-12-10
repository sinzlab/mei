FROM sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

RUN python3 -m pip install nnfabrik neuralpredictors

WORKDIR /src/mei
COPY . .
RUN python3 -m pip install -e .
