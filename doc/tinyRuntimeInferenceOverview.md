
# tinyRuntime Inference Overview

tinyRuntime is a lightweight ML framework designed for efficient execution of machine learning models, especially for quantized models. The inference is implemented in C and optimized for speed and size, making it ideal for scenarios where fast and small inference is critical. The primary goal of tinyRuntime is to provide a minimalistic yet effective runtime environment for running machine learning models with low overhead. Currently, tinyRuntime supports only ResNet18, 34 and 50.

tinyRuntime can be compiled and run with various configurations and optimizations. This overview provides details on how to compile and execute tinyRuntime with different settings.

## Compiling tinyRuntime

To compile tinyRuntime, run `make compile` command with environment variables to specify configurations. This will then generate two binaries `run` (vanilla tinyRuntime) and `runq` (quantization tinyRuntime).

- **Configuration Options**
   - **`BLAS=ON`**: Enables optimized functions using [BLIS](https://github.com/flame/blis) and [oneDNN](https://oneapi-src.github.io/oneDNN/) library. The default is `OFF`.
   - **`QUANT_TYPE=DQ` or `SQ`**: Specifies the quantization type. The default is `SQ`.
     - `DQ` for Dynamic Quantization.
     - `SQ` for Static Quantization.
   - **`STATIC=ON`**: Compiles tinyRuntime statically. The default is `OFF`. If using `STATIC=ON` with `BLAS=ON`, ensure that oneDNN and BLAS are installed statically. For static installation of BLAS libraries, refer to `install_static_blas.sh`.

Configurations will select the appropriate C files to be included in the inference.
Example compiles:
```bash
make compile
```
Output:
```bash
clang -Os -Wall run.c func_common.c func.c -o run -lm -lomp -fopenmp
clang -Os -Wall run.c func_common.c func_sq.c -o runq -lm -lomp -fopenmp
```
```bash
make compile BLAS=ON
```
Output:
```bash
clang -Os -Wall run.c func_common.c func_blis.c -o run -lblis -ldnnl -lm -lomp -fopenmp
clang -Os -Wall run.c func_common.c func_sq_onednn.c -o runq -lblis -ldnnl -lm -lomp -fopenmp
```

## Running tinyRuntime

To run tinyRuntime, you need to provide the paths to your model and data files. The execution will output the accuracy based on the provided data.

For running the vanilla tinyRuntime, use the following command format:
```bash
./run path/to/model path/to/data1 path/to/data2 ...
```
Similartly, for running the quantization tinyRuntime, run:
```bash
./runq path/to/model path/to/data1 path/to/data2 ...
```

By default, tinyRuntime will output the accuracy of the model based on the provided data. If you want to obtain the output of the model itself instead of the accuracy, append test after the command:
```bash
./run test path/to/model path/to/data1 path/to/data2 ...
```

tinyRuntime also supports batch processing. To specify the batch size for processing, use the BS environment variable when running tinyRuntime:
```bash
BS=8 ./run path/to/model path/to/data1 path/to/data2 ...
```

If tinyRuntime encounters errors such as segmentation faults, please first verify that you are using the correct model for your tinyRuntime. Common issues arise from model mismatches:
- Vanilla Model: Use only with vanilla tinyRuntime.
- Statically Quantized Model: Use only with static quantization tinyRuntime.
- Dynamically Quantized Model: Use only with dynamic quantization tinyRuntime.
Ensuring that the model matches the type of tinyRuntime you are using can resolve many common errors.
FIXME: Adding model type information (e.g., vanilla, dq, or sq) to the model file when exporting, and reading it in tinyRuntime, can help prevent these errors and make troubleshooting easier.

## Testing tinyRuntime

To test functions, you can use 
```bash
make ut
```
and then run
```bash
./ut
```
Similar to compiling inferences, you can select the function files to use by setting environment variables. Note that one unintuitive aspect of testing is that setting `QUANT_TYPE=DQ` will test both vanilla and dynamic quantization functions, while setting `QUANT_TYPE=SQ` will test only static quantization functions. Additionally, some of the unit tests may be overly complex, so you might consider simplifying them where possible.
