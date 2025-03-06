# TurboMind Quantization Plugin for vLLM

TurboMind is a high performance inference engine from InternLM Team, which is also the backend for [LMDeploy](https://github.com/InternLM/lmdeploy). [TurboMind](https://github.com/InternLM/turbomind) is the single kernel library for their quantized kernel. 

TurboMind has the ability to serve AWQ model on Volta GPUs (V100), which is the main reason I built this plugin. It also shows great performance on throughput.


::: warning

I'm not familiar with kernel development, most of the code here is migrated from the [Pr of SGLang](https://github.com/sgl-project/sglang/pull/2900) to vLLM, and it works fine on my device.
However, I haven't tested the accuracy on the model, please use it with caution.


# Installation

```bash
pip install git+https://github.com/liuyanyi/turbomind_vllm_plugin.git
```

# Usage

Since this package is a general plugin for vLLM, it will be loaded automatically when you run the vLLM OpenAI Server or `LLM` class.

But their's a small problem on vLLM cli args to use this plugin out of the box. I'm making a simple pr.

# Performance Test

This is not a formal test, just a simple comparison between TurboMind and other kernel.


benchmark command:
```bash
python3 benchmarks/benchmark_throughput.py --backend=vllm --model /large-storage/model/Qwen2.5/qwen/qwq-32b-awq/ -q {} --input-len 1024 --output-len 128 --num-prompts=250
```
The only difference is -q flag

A100 TP1 Test

| -q            | Device   | TP  | Throughput | Total Tokens/s | Output Tokens/s |
| ------------- | -------- | --- | ---------- | -------------- | --------------- |
| awq           | A100 80G | 1   | 1.77       | 2042.61        | 226.96          |
| awq_marlin    | A100 80G | 1   | 2.00       | 2298.71        | 255.41          |
| awq_turbomind | A100 80G | 1   | 2.48       | 2852.13        | 316.90          |

A100 TP2 Test

AWQ Throughput: 1.96 requests/s, 2263.15 total tokens/s, 251.46 output tokens/s
awq_marlin Throughput: 2.06 requests/s, 2377.60 total tokens/s, 264.18 output tokens/s
Throughput: 2.31 requests/s, 2662.31 total tokens/s, 295.81 output tokens/s

| -q            | Device   | TP  | Throughput | Total Tokens/s | Output Tokens/s |
| ------------- | -------- | --- | ---------- | -------------- | --------------- |
| awq           | A100 80G | 2   | 1.96       | 2263.15        | 251.46          |
| awq_marlin    | A100 80G | 2   | 2.06       | 2377.60        | 264.18          |
| awq_turbomind | A100 80G | 2   | 2.31       | 2662.31        | 295.81          |

Note: I'm using A100 PCIe 80G, the throughput is limited. It's also weird for tp2 slower than tp1 when using turbomind.

V100 TP2 Test

TODO


# Reference

- [TurboMind](https://github.com/InternLM/turbomind)
- [LMDeploy](https://github.com/InternLM/lmdeploy)
- [SGLang pr#2900](https://github.com/sgl-project/sglang/pull/2900)
