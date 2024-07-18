# MoD-Experiment

## Set up environment
- [Mixture-of-depths](https://github.com/kevin1010607/Mixture-of-depths)
- [alignment-handbook](https://github.com/kevin1010607/alignment-handbook)
- [lm-evaluation-harness](https://github.com/kevin1010607/lm-evaluation-harness)

```bash
# Install git large file ststem
sudo apt update
sudo apt install git-lfs
git lfs install

# Install python package
pip install -r requirements.txt

# Login to huggingface
huggingface-cli login --token {your_token}
huggingface-cli whoami

# Install customized python package
git clone https://github.com/kevin1010607/Mixture-of-depths.git
git clone https://github.com/kevin1010607/alignment-handbook.git
git clone https://github.com/kevin1010607/lm-evaluation-harness.git
pip install -e Mixture-of-depths
pip install -e alignment-handbook
pip install -e lm-evaluation-harness
```

## Train
```bash
cd alignment-handbook

# sft
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_sft.py \
    recipes/llama2/sft/config_full.yaml \
    | tee sft.log

# dpo
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_dpo.py \
    recipes/llama2/dpo/config_full.yaml \
    | tee dpo.log
```

## Evaluate
```bash
# nq_open
accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=kevin1010607/llama2-7b-mod-sft-full-3,tokenizer=meta-llama/Llama-2-7b-hf" \
    --tasks nq_open \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path nq_open-0

# mmlu
accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=kevin1010607/llama2-7b-mod-sft-full-3,tokenizer=meta-llama/Llama-2-7b-hf" \
    --tasks mmlu \
    --batch_size 1 \
    --num_fewshot 5 \
    --output_path mmlu-5
```