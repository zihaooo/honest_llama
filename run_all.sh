set -euo pipefail

source /home/zihaooo/miniconda3/bin/activate iti


model_names=("llama_7B" "llama2_chat_7B" "llama3_8B_instruct")
dataset_names=("tqa_mc2" "openbookqa_mc2" "mmlu_mc2")

for i in "${!model_names[@]}"; do
    model_name=${model_names[$i]}
    for j in "${!dataset_names[@]}"; do
      dataset_name=${dataset_names[$j]}
      # Run the activation script
      cd /scratch/mcity_project_root/mcity_project/zihaooo/iti/get_activations
      python get_activations.py --model_name $model_name --dataset_name $dataset_name
      python get_activations.py --model_name $model_name --dataset_name ${dataset_name/_mc2/_gen_end_q}
      # run validation
      cd /scratch/mcity_project_root/mcity_project/zihaooo/iti/validation
      python validate_2fold.py --model_name $model_name --dataset_name $dataset_name --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --instruction_prompt default --seed 1
  done
done

cd /scratch/mcity_project_root/mcity_project/zihaooo/iti/validation
python plot_head_acc.py