FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

RUN apt update && apt install -y \
    libgmp3-dev gawk \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]