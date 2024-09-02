# Repository Overview

This document provides a summary of each directory and file in this repository. It aims to help you understand the structure and purpose of the different components of the project.

## Directory Structure

## `.github/workflows/`
- **Description:** Contains GitHub Actions workflow files that automate various CI/CD tasks related to the project. Each workflow defines a set of automated processes triggered by specific events or conditions.
- **Files:**
  - `inference-test.yml`: Tests the validity of tinyRuntime inferences. It ensures that the results of these inferences are accurate and meet predefined criteria.
  - `benchmark.yml`: Measures and benchmarks the execution time of inferences, then uploads the performance metrics to the `ninjalabo.github.io` repository for analysis and visualization.
  - `compiler-test.yml`: Verifies that Docker images for the compiler in tinyMLaaS are successfully built according to the `Dockerfile`. It ensures that the images are created correctly and function as expected.

## `.vscode/`
- **Description:** Contains configuration files for Visual Studio Code (VSCode) to customize the development environment for the project. This ensures a consistent coding experience for all contributors. We follow coding style introduced in `CodingStyleSettings_VSCode.md`.
- **Files:**
  - `settings.json`: Configures project-specific settings for VSCode, such as code formatting preferences, linting rules, and editor behavior. You can use this file directly or copy necessary content to your own vscode settings. Please update this file as needed.

### `doc/`
- **Description:** Documentation files for the project.
- **Files:**
  - `CodingStyleSettings_VSCode.md`: Introduce coding style we use in C.
  - `RepositoryOverview.md`: Introduce repository structure and files.
  - `tinyRuntimeOverview.md`: Introduce how to compile and run tinyRuntime.

### `test/`
- **Description:** Contains files used in automatic tests run by GitHub Actions workflow. 
- **Files:**
  - `imagenette2-320/data/5/4.bin`: A test image from Imagenette data set.
  - `runtime_info.json`: File required in `compiler-test.yml`.

## Python files

- `benchmark.ipynb/py`: Scripts for benchmarking, utilized in the benchmark.yml GitHub Actions workflow.
- `export_model_vanilla.py`: Exports a vanilla model file for tinyRuntime, used in the `Dockerfile`.
- `export_model_dq.py`: Exports dynamic quantization model file for tinyRuntime. Used in `Dockerfile`.
- `export_model_sq.py`: Exports static quantization model file for tinyRuntime. Used in `Dockerfile`.
- `export.ipynb/py`: Contains the source code for functions that quantize and export PyTorch models for tinyRuntime.
- `nbexport.ipynb/py`: Scripts for converting Jupyter notebooks (.ipynb) to Python files (.py).
- `test_all.ipynb/py`: Scripts for testing inferences. Used in `inference-test.yml` GitHub Actions workflow.
- `test_dataset_generator.ipynb`: Notebook for generating test data set for Imagenette data set.
- `train.ipynb/py`: Scripts for training PyTorch models.

## C files

- `config_common.h`: Defines structs that are common across all inference types.
- `config_vanilla.h`: Defines structs used exclusively in the vanilla tinyRuntime.
- `config_dq.h`: Defines structs used exclusively in the dynamic quantization tinyRuntime.
- `config_sq.h`: Defines structs used exclusively in the static quantization tinyRuntime.
- `func_common.h`: Contains declarations for functions implemented in `func_common.c`.
  - `func_common.c`: Implements functions common to all inference types.
- `func.h`: Contains declarations for functions implemented in `func.c` and `func_blis.c`.
  - `func.c`: Implements functions used exclusively in the vanilla tinyRuntime.
  - `func_blis.c`: Optimized version of `func.c` that leverages the BLIS library.
- `func_q.h`: Contains declarations for functions implemented in `func_dq.c`, `func_dq_onednn.c`, `func_sq.c` and `func_sq_onednn.c`.
  - `func_dq.c`: Implements functions used exclusively in the dynamic quantization tinyRuntime.
  - `func_dq_onednn.c`: Optimized version of `func_dq.c` that leverages the oneDNN library.
  - `func_sq.c`: Implements functions used exclusively in the static quantization tinyRuntime.
  - `func_sq_onednn.c`: Optimized version of `func_sq.c` that leverages the oneDNN library.
- `run.c`: Vanilla tinyRuntime inference
- `runq.c`: Quantization tinyRuntime inference
- `test_func.c`: Contains unit tests for functions defined in the `func.*` files.
- `test_speed.c`: Benchmarks the speed of functions. (Outdated)
- `utmain.c`: Required for using CppUTest.

## Other files
- `.gitignore`: Specifies files and directories that Git should ignore.
- `Dockerfile`: Contains commands to create Docker images for tinyMLaaS compiler.
- `Makefile`: Provides a set of commands for tasks such as compiling tinyRuntime.
- `install_static_blas.sh`: Script to statically install BLIS and oneDNN, necessary for compiling tinyRuntime with static linking.
- `README.md`: Provides an overview of the project.
- `requirements.txt`: Lists the Python packages needed in the GitHub Actions workflows.
