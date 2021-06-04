
#### REAL RUNS ON BENCHMARK DATA
mkdir -p short_runs

# Run the conll languages
for lang in eng deu esp ned
do
    echo
    echo DO $lang

    batch_size=8
    model_name="bert-base-multilingual-cased"
    if [ $lang == "eng" ]
    then
        model_name="roberta-base"
    fi 

    echo "python main.py -dd data/conll2003/${lang}_P-1000 -cd save_${lang}_P-1000_all -rd resource --model-name $model_name --batch_size $batch_size -en 20 &> runs/21-06-01_run_${lang}_P-1000_all.log"
    # python main.py -dd data/conll2003/${lang}_P-1000_short -cd save_${lang}_P-1000_short -rd resource --model-name $model_name --batch_size $batch_size -en 50 &> runs/21-06-01_run_${lang}_P-1000_short.log

done


# Run the ontonotes languages
for lang in english chinese arabic
do
    echo
    echo DO $lang

    batch_size=8
    model_name="bert-base-multilingual-cased"
    if [ $lang == "english" ]
    then 
        model_name="roberta-base"
    fi 
    echo "python main.py -dd data/ontonotes5/${lang}_P-1000 -cd save_${lang}_P-1000_all -rd resource --model-name $model_name --batch_size $batch_size -en 20 &> runs/21-06-01_run_${lang}_P-1000_all.log"
    # python main.py -dd data/ontonotes5/${lang}_P-1000_short -cd save_${lang}_P-1000_short -rd resource --model-name $model_name --batch_size $batch_size -en 50 &> runs/21-06-01_run_${lang}_P-1000_short.log


done