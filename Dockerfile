FROM sinzlab/pytorch:v1.3.1-cuda10.1-dj0.12.4 as intermediate

WORKDIR /
COPY .ssh/id_rsa .
RUN eval $(ssh-agent) && \
    ssh-add id_rsa && \
    ssh-keyscan -H github.com >> /etc/ssh/ssh_known_hosts && \
    git clone git@github.com:sinzlab/nnfabrik /src/nnfabrik && \
    git clone git@github.com:sinzlab/ml-utils /src/ml-utils



FROM sinzlab/pytorch:v1.3.1-cuda10.1-dj0.12.4

COPY --from=intermediate /src /src
WORKDIR /src
RUN python3 ./nnfabrik/setup.py install && \
    python3 ./ml-utils/setup.py install

WORKDIR /src/mei
COPY . .
RUN python3 setup.py develop