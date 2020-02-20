FROM sinzlab/pytorch:v3.8-torch1.4.0-cuda10.1-dj0.12.4 as intermediate

WORKDIR /
COPY .ssh/id_rsa .
# Need to change the permissions of the private key because otherwise they are too open if the key is copied from
# Windows
RUN chmod 400 id_rsa && \
    eval $(ssh-agent) && \
    ssh-add id_rsa && \
    ssh-keyscan -H github.com >> /etc/ssh/ssh_known_hosts && \
    git clone git@github.com:sinzlab/nnfabrik /src/nnfabrik && \
    git clone git@github.com:sinzlab/ml-utils /src/ml-utils



FROM sinzlab/pytorch:v3.8-torch1.4.0-cuda10.1-dj0.12.4

COPY --from=intermediate /src /src
WORKDIR /src
RUN python3 -m pip install ./nnfabrik && \
    python3 -m pip install ./ml-utils

RUN apt-get update && apt-get install -y nodejs npm
RUN jupyter labextension install jupyterlab_vim

WORKDIR /src/mei
COPY . .
RUN python3 -m pip install -e .
