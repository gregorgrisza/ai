# Inferencing fine-tuned model

## CPP

### Configuration

- sentencepiece installation

    _Mac_
    ```bash
    git clone https://github.com/google/sentencepiece.git
    cd sentencepiece/
    mkdir build
    cmake ..
    cd build/
    cmake ..
    make -j $(nproc)
    sudo make install 
    sudo update_dyld_shared_cache

    ```
    See: [Build and install SentencePiece command line tools from C++ source](https://github.com/google/sentencepiece#build-and-install-sentencepiece-command-line-tools-from-c-source)