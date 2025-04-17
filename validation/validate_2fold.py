import torch
from einops import rearrange
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append('../')
from utils.utils import alt_tqa_evaluate, get_interventions_dict, \
    get_top_heads, get_separated_activations, get_com_directions, get_special_directions, get_matrix_directions

HF_NAMES = {
    # Base models
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',

    # HF edited models (ITI baked-in)
    'honest_llama_7B': 'jujipotle/honest_llama_7B', # Heads=48, alpha=15
    # 'honest_llama2_chat_7B': 'likenneth/honest_llama2_chat_7B', # Heads=?, alpha=?
    'honest_llama2_chat_7B': 'jujipotle/honest_llama2_chat_7B', # Heads=48, alpha=15
    'honest_llama2_chat_13B': 'jujipotle/honest_llama2_chat_13B', # Heads=48, alpha=15
    'honest_llama2_chat_70B': 'jujipotle/honest_llama2_chat_70B', # Heads=48, alpha=15
    'honest_llama3_8B_instruct': 'jujipotle/honest_llama3_8B_instruct', # Heads=48, alpha=15
    'honest_llama3_70B_instruct': 'jujipotle/honest_llama3_70B_instruct', # Heads=48, alpha=15
    # Locally edited models (ITI baked-in)
    'local_llama_7B': 'results_dump/edited_models_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_7B': 'results_dump/edited_models_dump/llama2_chat_7B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_13B': 'results_dump/edited_models_dump/llama2_chat_13B_seed_42_top_48_heads_alpha_15',
    'local_llama2_chat_70B': 'results_dump/edited_models_dump/llama2_chat_70B_seed_42_top_48_heads_alpha_15',
    'local_llama3_8B_instruct': 'results_dump/edited_models_dump/llama3_8B_instruct_seed_42_top_48_heads_alpha_15',
    'local_llama3_70B_instruct': 'results_dump/edited_models_dump/llama3_70B_instruct_seed_42_top_48_heads_alpha_15'
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix to model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)
    parser.add_argument('--use_special_direction', action='store_true', default=False)
    parser.add_argument('--use_mat_direction', action='store_true', default=False)
    parser.add_argument('--use_existed_direction', action='store_true', default=False)
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset_name == 'tqa_mc2':
        dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
        df = pd.read_csv('../TruthfulQA.csv')
        assert list(dataset['question']) == list(df["Question"])
    elif args.dataset_name == 'openbookqa_mc2':
        df = pd.read_csv('../OpenBookQA.csv')
    elif args.dataset_name == 'mmlu_mc2':
        df = pd.read_csv('../MMLU.csv')

    
    # get two folds using numpy
    fold_idxs = np.array_split(np.arange(len(df)), args.num_fold)

    # create model
    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # load activations 
    head_wise_activations = np.load(f"../features/{args.model_name}_{args.dataset_name}_head_wise.npy")
    labels = np.load(f"../features/{args.model_name}_{args.dataset_name}_labels.npy")
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    # tuning dataset: no labels used, just to get std of activations along the direction
    activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
    tuning_activations = np.load(f"../features/{args.model_name}_{activations_dataset}_head_wise.npy")
    tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
    tuning_labels = np.load(f"../features/{args.model_name}_{activations_dataset}_labels.npy")

    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations, dataset_name=args.dataset_name)
    # run k-fold cross validation
    results = []
    for i in range(args.num_fold):

        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # save train and test splits
        df.iloc[train_set_idxs].to_csv(f"splits/fold_{i}_train_seed_{args.seed}.csv", index=False)
        df.iloc[val_set_idxs].to_csv(f"splits/fold_{i}_val_seed_{args.seed}.csv", index=False)
        df.iloc[test_idxs].to_csv(f"splits/fold_{i}_test_seed_{args.seed}.csv", index=False)

        # get directions
        if args.use_center_of_mass:
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs,
                                                separated_head_wise_activations, separated_labels)
        elif args.use_special_direction:
            com_directions = get_special_directions(num_layers, num_heads, train_set_idxs, val_set_idxs,
                                                    separated_head_wise_activations, separated_labels, df)
        elif args.use_mat_direction:
            com_directions = get_matrix_directions(num_layers, num_heads, train_set_idxs, val_set_idxs,
                                                   separated_head_wise_activations, separated_labels)
        else:
            com_directions = None

        print("Finished computing com_directions of shape", com_directions.shape)
        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
        print("Heads intervened: ", sorted(top_heads))
    
        interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, args.use_mat_direction, args.use_special_direction, com_directions)
        print("Finished computing interventions dict")

        if args.dataset_name == 'tqa_mc2':
            if args.use_center_of_mass:
                _filename = 'direction_from_truthfulqa_com.npy'
            elif args.use_special_direction:
                _filename = 'direction_from_truthfulqa_special.npy'
            else:
                _filename = 'direction_from_truthfulqa_mat.npy'
            np.save(_filename, interventions)
            print(f"Saved com_directions to {_filename}")
        if args.use_existed_direction and args.dataset_name != 'tqa_mc2':
            if args.use_center_of_mass:
                _filename = 'direction_from_truthfulqa_com.npy'
            elif args.use_special_direction:
                _filename = 'direction_from_truthfulqa_special.npy'
            else:
                _filename = 'direction_from_truthfulqa_mat.npy'
            interventions = np.load(_filename)
            print(f"Use direction from TruthfulQA dataset: {_filename}")


        def lt_modulated_vector_add(_head_output, layer_name, start_edit_location='lt', prompt_encoding=None):

            head_output = _head_output.detach().type(torch.float32)
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            layer = int(layer_name.split('.')[2])
            # print("head output shape", head_output.shape)

            # Get actual runtime head dimension
            runtime_head_dim = head_output.shape[-1]

            if prompt_encoding is not None:  # use_special_direction
                assert prompt_encoding.shape == (384,)
                prompt_encoding = torch.FloatTensor(prompt_encoding).to(head_output.device.index).reshape(-1, 384)
            # print("Prompt encoding", prompt_encoding.shape)

            for head, direction, proj_val_std in interventions[layer_name]:
                if len(direction.shape) == 2:  # use_mat_direction or use_special_direction
                    activations = torch.FloatTensor(tuning_activations[:, layer, head, :]).to(
                        head_output.device.index)  # batch_all x 128
                    assert (proj_val_std is None)
                    direction = torch.FloatTensor(direction).to(head_output.device.index)  # 128 x 384

                    if start_edit_location == 'lt':
                        if prompt_encoding is None:
                            direction_to_add = head_output[:, -1, head, :] @ direction.T  # batch x 128
                        else:  # use_special_direction
                            # uses batch size = 1
                            direction_to_add = prompt_encoding @ direction.T  # 1 x 128
                        direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add, axis=1).reshape(-1, 1)

                        # compute stddev online
                        proj_vals = activations @ direction_to_add.T  # batch_all x batch
                        proj_val_std = torch.std(proj_vals, axis=0).reshape(1, -1)  # batch x 1
                        # print("proj_val_std has shape", proj_val_std.shape)

                    else:
                        if prompt_encoding is None:
                            direction_to_add = torch.einsum('bij,jk->bik',
                                                            head_output[:, start_edit_location:, head, :], direction.T)
                            direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add, axis=2)[:, :,
                                                                  None]  # batch x location_indices x 128
                        else:
                            direction_to_add = prompt_encoding @ direction.T  # 1 x 128
                            direction_to_add = direction_to_add.unsqueeze(1).repeat(1, head_output.shape[
                                1] - start_edit_location, 1)  # 1 x location_indices, 128
                            direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add, axis=2)[:, :,
                                                                  None]  # batch x location_indices x 128

                        # compute stddev online
                        proj_vals = torch.einsum('Bk,bik->Bbi', activations, direction_to_add)
                        proj_val_std = torch.std(proj_vals, axis=0)[:, :, None]  # batch x location_indices x 1
                        # print("proj_val_std has shape", proj_val_std.shape)

                    proj_val_std = torch.Tensor(proj_val_std).to(head_output.device.index)
                    if start_edit_location == 'lt':
                        head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                    else:
                        head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
                else:
                    assert (proj_val_std is not None)
                    direction_to_add = torch.FloatTensor(direction).to(head_output.device.index)  # 128 x 1
                    if start_edit_location == 'lt':
                        head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                    else:
                        head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output.type(torch.float16)

        filename = f'{args.model_prefix}{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'

        if args.use_center_of_mass:
            filename += '_com'
        if args.use_random_dir:
            filename += '_random'
        # if args.use_honest:
        #     filename = 'honest_' + filename
        if args.use_special_direction:
            filename += '_special'
        if args.use_mat_direction:
            filename += '_mat'
        metric_names    = ['mc']
        curr_fold_results = alt_tqa_evaluate(
            models={args.model_name: model},
            metric_names=metric_names,
            input_path=f'splits/fold_{i}_test_seed_{args.seed}.csv',
            output_path=f'results_dump/answer_dump/{filename}.csv',
            summary_path=f'results_dump/summary_dump/{filename}.csv',
            device="cuda", 
            interventions=interventions, 
            intervention_fn=lt_modulated_vector_add, 
            instruction_prompt=args.instruction_prompt,
            judge_name=args.judge_name, 
            info_name=args.info_name,
            use_special_direction=args.use_special_direction
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)
    if 'info' not in metric_names and 'judge' not in metric_names:
        print(f'alpha: {args.alpha}, heads: {args.num_heads}, MC1 Score: {final[0]}, MC2 Score: {final[1]}, CE Loss: {final[2]}, KL wrt Original: {final[3]}')
    else:
        print(f'alpha: {args.alpha}, heads: {args.num_heads}, True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()
