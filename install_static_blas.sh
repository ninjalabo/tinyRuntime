# Script to install BLIS and oneDNN libraries statically

sudo apt-get update
sudo apt-get install -y build-essential cmake
# Install BLIS library statically
git clone https://github.com/flame/blis.git
cd blis && ./configure --enable-static generic && make -j && make install
cd ..

# Install oneDNN library statically for linux
# FIX: link arm compute libraries when installing oneDNN on arm
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN
mkdir -p build && cd build
cmake .. -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_DOC=OFF -DONEDNN_BUILD_EXAMPLES=OFF \
         -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_GRAPH=OFF -DONEDNN_ENABLE_JIT_PROFILING=OFF \
	 -DONEDNN_ENABLE_ITT_TASKS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE
make -j # specify the number of threads to use if you run into memory issues
cmake --build . --target install

# After installing the libraries, you can compile run.c and runq.c statically by specifying STATIC=ON
