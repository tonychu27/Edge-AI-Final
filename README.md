# Edge-AI-Final

Team: 5

## Environment Setup

For setting up `nvcc`
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run
```

Add Path:
```bash
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

And the requirements package
```bash
pip install torch
pip install datasets
pip install vllm
pip install --upgrade accelerate
pip install gptqmodel
pip install threadpoolctl
pip install device_smi

pip install -r requirements.txt
```

## Evaluation
Run the script to evaluate the throughput and perplexity.

For evaluating perplexity
```bash
python3 result_perplexity.py
```

For evaluating throughput
```bash
python3 result_throughput.py
```