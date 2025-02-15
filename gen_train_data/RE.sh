python Geneticprompt.py --dataset=chemprot --batch_size=20 --n_sample=1500 --top_p=1.0 \
 --temperature=1.0 --output_dir=../VLLM-Synthetic_data/GP --model_name=meta-llama/Llama-3.3-70B-Instruct \
 --parents="active"

python Geneticprompt.py --dataset=ddi --batch_size=20 --n_sample=1500 --top_p=1.0 \
 --temperature=1.0 --output_dir=../VLLM-Synthetic_data/GP --model_name=meta-llama/Llama-3.3-70B-Instruct \
 --parents="active"

python Geneticprompt.py --dataset=conll04 --batch_size=20 --n_sample=1500 --top_p=1.0 \
 --temperature=1.0 --output_dir=../VLLM-Synthetic_data/GP --model_name=meta-llama/Llama-3.3-70B-Instruct \
 --parents="active"

python Geneticprompt.py --dataset=semeval2010 --batch_size=20 --n_sample=1500 --top_p=1.0 \
 --temperature=1.0 --output_dir=../VLLM-Synthetic_data/GP --model_name=meta-llama/Llama-3.3-70B-Instruct \
 --parents="active"
