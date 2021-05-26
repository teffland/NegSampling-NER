# # Setup env
# conda create negsampling-ner python=3.6
# conda activate negsampling-ner
# pip install pytorch_pretrained_bert allennlp==1.2.1


# # Convert data to their format



# python convert_data.py --in-path ../data/conll2003/eng/entity.train.jsonl --out-path data/conll2003/eng/train.json
# python convert_data.py --in-path ../data/conll2003/eng/entity.dev.jsonl --out-path data/conll2003/eng/dev.json
# python convert_data.py --in-path ../data/conll2003/eng/entity.test.jsonl --out-path data/conll2003/eng/test.json

# python convert_data.py --in-path ../data/conll2003/eng/entity.train_P-1000.jsonl --out-path data/conll2003/eng_P-1000/train.json
# python convert_data.py --in-path ../data/conll2003/eng/entity.dev.jsonl --out-path data/conll2003/eng_P-1000/dev.json
# python convert_data.py --in-path ../data/conll2003/eng/entity.test.jsonl --out-path data/conll2003/eng_P-1000/test.json

# python convert_data.py --in-path ../data/conll2003/eng/entity.train_r0.5_p0.9.jsonl --out-path data/conll2003/eng_r0.5_p0.9/train.json
# python convert_data.py --in-path ../data/conll2003/eng/entity.dev.jsonl --out-path data/conll2003/eng_r0.5_p0.9/dev.json
# python convert_data.py --in-path ../data/conll2003/eng/entity.test.jsonl --out-path data/conll2003/eng_r0.5_p0.9/test.json

# # Setup resources
# mkdir save
# mkdir -p resource/roberta-base

# # Make sure we have our bert model with their format
# curl https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json --output resource/roberta-base/vocab.json
# python convert_roberta_vocab.py

# ## Run a first test
# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py -dd data/conll2003/eng -cd save -rd resource --model-name roberta-base --batch_size 8 &> 21-05-11_run_eng_full.log &
# nohup python main.py -dd data/conll2003/eng_P-1000 -cd save -rd resource --model-name roberta-base --batch_size 8 &> 21-05-11_run_eng-c_P-1000.log &
# nohup python main.py -dd data/conll2003/eng_r0.5_p0.9 -cd save -rd resource --model-name roberta-base --batch_size 8 &> 21-05-11_run_eng-c_r0.5_p0.9.log &

#### REAL RUNS ON BENCHMARK DATA
mkdir -p runs

# Run the conll languages
# for lang in eng deu esp ned
# for lang in deu esp ned
# do
lang=ned
echo
echo DO $lang
# python convert_data.py --in-path ../data/conll2003/$lang/entity.train.jsonl --out-path data/conll2003/$lang/train.json
# python convert_data.py --in-path ../data/conll2003/$lang/entity.dev.jsonl --out-path data/conll2003/$lang/dev.json
# python convert_data.py --in-path ../data/conll2003/$lang/entity.test.jsonl --out-path data/conll2003/$lang/test.json

# python convert_data.py --in-path ../data/conll2003/$lang/entity.train_P-1000.jsonl --out-path data/conll2003/${lang}_P-1000/train.json
# python convert_data.py --in-path ../data/conll2003/$lang/entity.dev.jsonl --out-path data/conll2003/${lang}_P-1000/dev.json
# python convert_data.py --in-path ../data/conll2003/$lang/entity.test.jsonl --out-path data/conll2003/${lang}_P-1000/test.json

# python convert_data.py --in-path ../data/conll2003/$lang/entity.train_r0.5_p0.9.jsonl --out-path data/conll2003/${lang}_r0.5_p0.9/train.json
# python convert_data.py --in-path ../data/conll2003/$lang/entity.dev.jsonl --out-path data/conll2003/${lang}_r0.5_p0.9/dev.json
# python convert_data.py --in-path ../data/conll2003/$lang/entity.test.jsonl --out-path data/conll2003/${lang}_r0.5_p0.9/test.json

batch_size=8
model_name="bert-base-multilingual-cased"
if [ $lang == "eng" ]
then
    model_name="roberta-base"
fi 
echo "python main.py -dd data/conll2003/$lang -cd save_${lang} -rd resource --model-name $model_name --batch_size $batch_size &> runs/21-05-17_run_${lang}_full.log"
python main.py -dd data/conll2003/$lang -cd save_${lang} -rd resource --model-name $model_name --batch_size $batch_size &> runs/21-05-17_run_${lang}_full.log
echo "python main.py -dd data/conll2003/${lang}_P-1000 -cd save_${lang}_P-1000 -rd resource --model-name $model_name --batch_size $batch_size -en 20 &> runs/21-05-17_run_${lang}_P-1000.log"
python main.py -dd data/conll2003/${lang}_P-1000 -cd save_${lang}_P-1000 -rd resource --model-name $model_name --batch_size $batch_size -en 20 &> runs/21-05-17_run_${lang}_P-1000.log
echo "python main.py -dd data/conll2003/${lang}_r0.5_p0.9 -cd save_${lang}_r0.5_p0.9 -rd resource --model-name $model_name --batch_size $batch_size -en 20 &> runs/21-05-17_run_${lang}_r0.5_p0.9.log"
python main.py -dd data/conll2003/${lang}_r0.5_p0.9 -cd save_${lang}_r0.5_p0.9 -rd resource --model-name $model_name --batch_size $batch_size -en 20 &> runs/21-05-17_run_${lang}_r0.5_p0.9.log

# done


# Run the ontonotes languages
# for lang in english chinese arabic
# do
lang=arabic
echo
echo DO $lang
# python convert_data.py --in-path ../data/ontonotes5/processed/$lang/train.jsonl --out-path data/ontonotes5/$lang/train.json
# python convert_data.py --in-path ../data/ontonotes5/processed/$lang/dev.jsonl --out-path data/ontonotes5/$lang/dev.json
# python convert_data.py --in-path ../data/ontonotes5/processed/$lang/test.jsonl --out-path data/ontonotes5/$lang/test.json

# python convert_data.py --in-path ../data/ontonotes5/processed/$lang/train_P-1000.jsonl --out-path data/ontonotes5/${lang}_P-1000/train.json
# python convert_data.py --in-path ../data/ontonotes5/processed/$lang/dev.jsonl --out-path data/ontonotes5/${lang}_P-1000/dev.json
# python convert_data.py --in-path ../data/ontonotes5/processed/$lang/test.jsonl --out-path data/ontonotes5/${lang}_P-1000/test.json

# python convert_data.py --in-path ../data/ontonotes5/processed/$lang/train_r0.5_p0.9.jsonl --out-path data/ontonotes5/${lang}_r0.5_p0.9/train.json
# python convert_data.py --in-path ../data/ontonotes5/processed/$lang/dev.jsonl --out-path data/ontonotes5/${lang}_r0.5_p0.9/dev.json
# python convert_data.py --in-path ../data/ontonotes5/processed/$lang/test.jsonl --out-path data/ontonotes5/${lang}_r0.5_p0.9/test.json

batch_size=8
model_name="bert-base-multilingual-cased"
if [ $lang == "english" ]
then 
    model_name="roberta-base"
fi 
echo "python main.py -dd data/ontonotes5/$lang -cd save_${lang} -rd resource --model-name $model_name --batch_size $batch_size &> runs/21-05-17_run_${lang}_full.log"
python main.py -dd data/ontonotes5/$lang -cd save_${lang} -rd resource --model-name $model_name --batch_size $batch_size &> runs/21-05-17_run_${lang}_full.log
echo "python main.py -dd data/ontonotes5/${lang}_P-1000 -cd save_${lang}_P-1000 -rd resource --model-name $model_name --batch_size $batch_size -en 20 &> runs/21-05-17_run_${lang}_P-1000.log"
python main.py -dd data/ontonotes5/${lang}_P-1000 -cd save_${lang}_P-1000 -rd resource --model-name $model_name --batch_size $batch_size -en 20 &> runs/21-05-17_run_${lang}_P-1000.log
echo "python main.py -dd data/ontonotes5/${lang}_r0.5_p0.9 -cd save_${lang}_r0.5_p0.9 -rd resource --model-name $model_name --batch_size $batch_size -en 20 &> runs/21-05-17_run_${lang}_r0.5_p0.9.log"
python main.py -dd data/ontonotes5/${lang}_r0.5_p0.9 -cd save_${lang}_r0.5_p0.9 -rd resource --model-name $model_name --batch_size $batch_size &> runs/21-05-17_run_${lang}_r0.5_p0.9.log


# done