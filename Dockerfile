# Multi-platform build can be easily done using Docker Desktop
# see (https://www.docker.com/blog/multi-arch-build-and-images-the-simple-way/)
# NOTE: To build multi-platform docker image, you should create a new builder instance that uses a driver
# supporting multi-platform builds, such as the docker-container driver. You can do this by running:
# docker buildx create --use --name mybuilder --driver docker-container
# docker buildx inspect mybuilder --bootstrap


# MODEL EXPORT COMPILER
# Common base stage for installing dependencies
FROM python:3.9-slim as compiler-export-base
WORKDIR /root
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastai fasterai

# Separate stage for vanilla model export
FROM compiler-export-base as compiler-export-vanilla
COPY export_model_vanilla.py export.py /root
CMD python export_model_vanilla.py
# To build and push the docker image, run:
# docker build --target compiler-export-vanilla --push --platform linux/amd64 -t ninjalabo/compiler-export-vanilla:latest .

# Separate stage for dq model export
FROM compiler-export-base as compiler-export-dq
COPY export_model_dq.py export.py /root
CMD python export_model_dq.py
# To build and push the docker image, run:
# docker build --target compiler-export-dq --push --platform linux/amd64 -t ninjalabo/compiler-export-dq:latest .

# Separate stage for sq model export
FROM compiler-export-base as compiler-export-sq
COPY export_model_sq.py export.py /root
CMD python export_model_sq.py
# To build and push the docker image, run:
# docker build --target compiler-export-sq --push --platform linux/amd64 -t ninjalabo/compiler-export-sq:latest .
