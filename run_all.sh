set -e

source /home/zihaooo/miniconda3/bin/activate iti

cd /scratch/mcity_project_root/mcity_project/zihaooo/iti/get_activations

#model_name=llama_7B
#model_name=llama2_chat_7B
model_name=llama3_8B_instruct
# model_name=gemma2_2B

python get_activations.py --model_name $model_name --dataset_name tqa_mc2
python get_activations.py --model_name $model_name --dataset_name tqa_gen_end_q

cd /scratch/mcity_project_root/mcity_project/zihaooo/iti/validation

python validate_2fold.py --model_name $model_name --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default --seed 1
python validate_2fold.py --model_name $model_name --num_heads 1 --alpha 0 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default --seed 1
python validate_2fold.py --model_name $model_name --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default --seed 2
python validate_2fold.py --model_name $model_name --num_heads 1 --alpha 0 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default --seed 2
python validate_2fold.py --model_name $model_name --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default --seed 3
python validate_2fold.py --model_name $model_name --num_heads 1 --alpha 0 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default --seed 3