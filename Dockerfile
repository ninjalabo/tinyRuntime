FROM ubuntu
COPY run resnet18.bin test/data/imagenette2/val_transformed/0/113 /root
WORKDIR /root
CMD ./run 18 resnet18.bin 113

