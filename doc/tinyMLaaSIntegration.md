
# Integrating tinyRuntime with tinyMLaaS

tinyRuntime is a lightweight machine learning framework designed to quantize models, export them, and efficiently run inferences on them. It is particularly well-suited for deployment in resource-constrained environments such as embedded systems and edge devices.

tinyMLaaS (tiny Machine Learning as a Service) is a service platform that leverages tinyRuntime to provide scalable, efficient machine learning services. To fully utilize the capabilities of tinyRuntime, it is necessary to integrate it seamlessly with tinyMLaaS.

This guide will walk you through the steps required to integrate tinyRuntime into tinyMLaaS.

In tinyMLaaS has 5 entities: Devices, Compilers, Installers, Datasets, Models, where TinyMLaaS will associate them each other and orchastrate devices eventually. tinyRuntime repository is responsible of Devices, Compilers, Datasets and Models.

## Devices

tinyMLaaS currently supports both ARM64 and AMD64 architectures. To use tinyRuntime within tinyMLaaS on these devices, you need to save the binary files of tinyRuntime inferences to the appropriate folders in the `https://github.com/ninjalabo/inferences` repository. These binaries must be statically compiled. For instructions on how to compile statically, refer to the `tinyRuntimeInferenceOverview.md`. Remember to enable BLAS libraries during compilation.

Note: At present, this process must be done manually. Ideally, it should be automated using GitHub Actions. However, this isn't feasible right now because installing oneDNN is a resource-intensive process. If possible, explore ways to install only the necessary dependencies from oneDNN to speed up the installation and reduce the binary size.

## Installers

Scripts for installing inferences and generating the Docker image for the tinyMLaaS installer can be found in the `https://github.com/ninjalabo/tinyMLaaS`.

## Compilers

Compiler updates can be easily managed by uploading Docker images specified in the `Dockerfile`. This process is not automated due to the time-consuming nature of pushing Docker images via GitHub Actions.

## Datasets

Datasets for tinyMLaaS are stored on Hugging Face. Test datasets can be generated using `test_dataset_generator.ipynb`, and should then be uploaded to Hugging Face.

## Models

Models are also stored on Hugging Face. They can be trained using the `train.ipynb` notebook. 
