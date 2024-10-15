



python RE-geneticprompt.py --dataset=chemprot --batch_size=2 --n_sample=700 --top_p=1.0 --temperature=1.0 --output_dir=../DATA/demoprompt/chemprot --model_name=gpt-3.5-turbo
python RE-geneticprompt.py --dataset=ddi --batch_size=2 --n_sample=800 --top_p=1.0 --temperature=1.0 --output_dir=../DATA/demoprompt/ddi           --model_name=gpt-3.5-turbo

python RE-geneticprompt.py --dataset=conll04 --batch_size=2 --n_sample=700 --top_p=1.0 --temperature=1.0 --output_dir=../DATA/geneticprompt/conll04 --model_name=gpt-3.5-turbo

python RE-geneticprompt.py --dataset=semeval2010 --batch_size=2 --n_sample=600 --top_p=1.0 --temperature=1.0 --output_dir=../DATA/geneticprompt/semeval2010 --model_name=gpt-3.5-turbo








