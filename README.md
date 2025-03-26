# TurboMind Quantization Plugin for vLLM

TurboMind is a high performance inference engine from InternLM Team, which is also the backend for [LMDeploy](https://github.com/InternLM/lmdeploy). [TurboMind](https://github.com/InternLM/turbomind) is the single kernel library for their quantized kernel. 

TurboMind has the ability to serve AWQ model on Volta GPUs (V100), which is the main reason I built this plugin. It also shows great performance on throughput.


> [!WARNING]  
> I'm not familiar with kernel development, most of the code here is migrated from the [Pr of SGLang](https://github.com/sgl-project/sglang/pull/2900) to vLLM, and it works fine on my device.
> However, I haven't tested the accuracy on the model, please use it with caution.


# Installation

First, clone this repository:

```bash
git clone https://github.com/liuyanyi/turbomind_vllm_plugin.git
```

remember to update the submodule:

```bash
git submodule update --init --recursive
```

## Step 1: Install TurboMind

### Choice 1: Install TurboMind from source

```bash
# bash setup_turbomind.sh <NUM_PROCESS> <NUM_THREADS>
bash setup_turbomind.sh 8 12
```

This will build and install the turbomind under `third_party/turbomind`.

### Choice 2: Install pre-built TurboMind

You can download the pre-built TurboMind package from the releases section of the repository. I built that on cuda 12 and python 3.10. Install the TurboMind package:

```bash
pip install turbomind-0.1.0-cp310-cp310-linux_x86_64.whl
```

### Choice 3: [NOT RECOMMENDED] Install TurboMind from pip

```bash
pip install turbomind
```

> [!WARNING]  
> Until now (20250326), turbomind on pypi is an old version, which lead to a higer memory usage (See https://github.com/InternLM/turbomind/issues/14), this version is runable but not recommended.


## Step 2: Install the Plugin

```bash
pip install .
```

the `turbomind_vllm_plugin` will be installed.


# Usage

Since this package is a general plugin for vLLM, it will be loaded automatically when you run the vLLM OpenAI Server or `LLM` class.

You can use vLLM>0.8.0 with this plugin, for example:

```bash
vllm serve <MODEL_TAG> -q awq_turbomind --enforce-eager
```

Since vLLM v1 engine is default to use, but this plugin is currently not compatible with torch.compile.
If you face the error like this or any error related to torch dynamo and torch.compile, 
You can use `--enforce-eager` in v1 engine to avoid this, or use v0 by set env `VLLM_USE_V1=0 `

```
[core.py:340]     raise Unsupported(msg, case_name=case_name)
[core.py:340] torch._dynamo.exc.Unsupported: call_method UserDefinedObjectVariable
```

# Performance Test

This is not a formal test, just a simple comparison between TurboMind and other kernel.


## Throughput Test on single A100

vLLM version: 0.8.2

commands:
```bash
python3 benchmarks/benchmark_throughput.py --backend=vllm --model /large-storage/model/Qwen2.5/qwen/qwq-32b-awq/ -q awq_turbomind --input-len 1024 --output-len 128 --num-prompts=1000 --enforce-eager --max-model-len 32768
python3 benchmarks/benchmark_throughput.py --backend=vllm --model /large-storage/model/Qwen2.5/qwen/qwq-32b-awq/ -q awq --input-len 1024 --output-len 128 --num-prompts=1000  --max-model-len 32768
python3 benchmarks/benchmark_throughput.py --backend=vllm --model /large-storage/model/Qwen2.5/qwen/qwq-32b-awq/ -q awq_marlin --input-len 1024 --output-len 128 --num-prompts=1000  --max-model-len 32768
```

TODO

## Serving Benchmark

Model: QwQ-32B-AWQ
vLLM version: 0.8.2

```
AWQ
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  633.82    
Total input tokens:                      1024000   
Total generated tokens:                  125336    
Request throughput (req/s):              1.58      
Output token throughput (tok/s):         197.75    
Total Token throughput (tok/s):          1813.36   
---------------Time to First Token----------------
Mean TTFT (ms):                          294287.98 
Median TTFT (ms):                        287592.85 
P99 TTFT (ms):                           592100.04 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          743.77    
Median TPOT (ms):                        789.47    
P99 TPOT (ms):                           832.61    
---------------Inter-token Latency----------------
Mean ITL (ms):                           744.99    
Median ITL (ms):                         950.71    
P99 ITL (ms):                            979.63    
==================================================
```

```
AWQ_MARLIN
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  583.36    
Total input tokens:                      1024000   
Total generated tokens:                  125336    
Request throughput (req/s):              1.71      
Output token throughput (tok/s):         214.85    
Total Token throughput (tok/s):          1970.19   
---------------Time to First Token----------------
Mean TTFT (ms):                          281009.08 
Median TTFT (ms):                        282297.23 
P99 TTFT (ms):                           564498.71 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          686.12    
Median TPOT (ms):                        742.45    
P99 TPOT (ms):                           780.88    
---------------Inter-token Latency----------------
Mean ITL (ms):                           686.61    
Median ITL (ms):                         957.65    
P99 ITL (ms):                            983.14    
==================================================
```


```
AWQ TurboMind
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  478.11    
Total input tokens:                      1024000   
Total generated tokens:                  125336    
Request throughput (req/s):              2.09      
Output token throughput (tok/s):         262.15    
Total Token throughput (tok/s):          2403.93   
---------------Time to First Token----------------
Mean TTFT (ms):                          229132.11 
Median TTFT (ms):                        230232.41 
P99 TTFT (ms):                           464637.11 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          540.20    
Median TPOT (ms):                        578.18    
P99 TPOT (ms):                           608.63    
---------------Inter-token Latency----------------
Mean ITL (ms):                           541.79    
Median ITL (ms):                         767.52    
P99 ITL (ms):                            797.92    
==================================================
```

# TODO

- [ ] Fix torch compile
- [ ] Update performance benchmarks

# Reference

- [TurboMind](https://github.com/InternLM/turbomind)
- [LMDeploy](https://github.com/InternLM/lmdeploy)
- [SGLang pr#2900](https://github.com/sgl-project/sglang/pull/2900)
