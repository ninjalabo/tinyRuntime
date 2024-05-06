# Multi-platform build can be easily done using Docker Desktop
# see (https://www.docker.com/blog/multi-arch-build-and-images-the-simple-way/)
# NOTE: To build multi-platform docker image, you should create a new builder instance that uses a driver
# supporting multi-platform builds, such as the docker-container driver. You can do this by running:
# docker buildx create --use --name mybuilder --driver docker-container
# docker buildx inspect mybuilder --bootstrap


# MODEL EXPORT COMPILER
FROM python:3.9-slim as compiler
WORKDIR /root
COPY prep_model.py export.py train.py /root
# TODO: In total over 1 GB, everything is not needed for sure (try `pip install --no-deps`)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastai fasterai
CMD python prep_model.py
# To build and push docker image, run: 
# docker buildx build --target compiler --push --platform linux/arm64,linux/amd64 -t ninjalabo/compiler-export-models:latest .

# COMPILING TINYRUNTIME INFERENCES STATICALLY
# By following instructions below, you can compile tinyRuntime inferences statically, and store it in https://github.com/ninjalabo/inferences
FROM ubuntu as test
WORKDIR /root
RUN apt-get update && apt-get install -y git make gcc python3
RUN git clone https://github.com/flame/blis.git
RUN cd blis && ./configure --enable-static generic && make -j && make install
# To build docker image run
# docker build -t test --target test
# After building the docker image, run in tinyRuntime repo
# docker run -v $(pwd):/root test make compile BLIS=1 ARCH=arm STATIC=1
