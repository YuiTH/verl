FROM verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2
RUN apt-get update && apt-get install -y openssh-server
COPY . /workspace/verl
RUN cd /workspace/verl && pip3 install --no-deps -e .