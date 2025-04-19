# -*- coding: utf-8 -*-
import numpy as np


def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"


def format_truthfulqa_end_q(question, choice, rand_question):
    return f"Q: {question} A: {choice} Q: {rand_question}"


def tokenized_tqa(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)):
            choice = choices[j]
            label = labels[j]
            prompt = format_truthfulqa(question, choice)
            if i == 0 and j == 0:
                print(prompt)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(label)

    return all_prompts, all_labels


def tokenized_tqa_gen_end_q(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        category = dataset[i]['category']
        rand_idx = np.random.randint(len(dataset))
        rand_question = dataset[rand_idx]['question']

        for j in range(len(dataset[i]['correct_answers'])):
            answer = dataset[i]['correct_answers'][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)

        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)

    return all_prompts, all_labels, all_categories


def format_openbookqa(question, choice):
    return f"Q: {question} A: {choice}"


def format_openbookqa_end_q(question, choice, rand_question):
    return f"Q: {question} A: {choice} Q: {rand_question}"


def add_ending(text, ending):
    if text[-1] != ending:
        text += ending
    return text


def tokenized_openbookqa(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = add_ending(dataset[i]['question_stem'], '?')
        choices = dataset[i]['choices']['text']
        _labels = dataset[i]['choices']['label']
        answer_key = dataset[i]['answerKey']


        assert len(choices) == len(_labels), (len(choices), len(_labels))

        for j in range(len(choices)):
            choice = add_ending(choices[j], '.')
            prompt = format_openbookqa(question, choice)
            if i == 0 and j == 0:
                print(prompt)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)

            label = 1 if _labels[j] == answer_key else 0
            all_labels.append(label)

    return all_prompts, all_labels


def tokenized_openbookqa_gen_end_q(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)):
        question = add_ending(dataset[i]['question_stem'], '?')
        rand_idx = np.random.randint(len(dataset))
        rand_question =  add_ending(dataset[rand_idx]['question_stem'], '?')
        answer_key = dataset[i]['answerKey']
        answer_by_key = dict(zip(dataset[i]['choices']['label'], dataset[i]['choices']['text']))

        for key, answer in answer_by_key.items():
            answer = add_ending(answer, '.')
            prompt = format_openbookqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)
            if key == answer_key:
                all_labels.append(1)
            else:
                all_labels.append(0)
            all_categories.append('')

    return all_prompts, all_labels, all_categories


def format_mmlu(question, choice):
    return f"Q: {question} A: {choice}"


def format_mmlu_end_q(question, choice, rand_question):
    return f"Q: {question} A: {choice} Q: {rand_question}"

def tokenized_mmlu(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = add_ending(dataset[i]['question'], '?')
        choices = dataset[i]['choices']
        _labels = list(range(len(choices)))
        answer_key = dataset[i]['answer']


        assert len(choices) == len(_labels), (len(choices), len(_labels))

        for j in range(len(choices)):
            choice = add_ending(choices[j], '.')
            prompt = format_mmlu(question, choice)
            if i == 0 and j == 0:
                print(prompt)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)

            label = 1 if _labels[j] == answer_key else 0
            all_labels.append(label)

    return all_prompts, all_labels


def tokenized_mmlu_gen_end_q(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)):
        question = add_ending(dataset[i]['question'], '?')
        rand_idx = np.random.randint(len(dataset))
        rand_question =  add_ending(dataset[rand_idx]['question'], '?')
        answer_key = dataset[i]['answer']
        answer_by_key = dict(zip(range(len(dataset[i]['choices'])), dataset[i]['choices']))

        for key, answer in answer_by_key.items():
            answer = add_ending(answer, '.')
            prompt = format_mmlu_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)
            if key == answer_key:
                all_labels.append(1)
            else:
                all_labels.append(0)
            all_categories.append('')

    return all_prompts, all_labels, all_categories
  
  
  
def format_arc(question, choice):
    return f"Q: {question} A: {choice}"

def tokenized_arc(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = add_ending(dataset[i]['question'], '?')
        choices = dataset[i]['choices']['text']
        labels = dataset[i]['choices']['label']
        answer_key = dataset[i]['answerKey']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)):
            choice = add_ending(choices[j], '.')
            prompt = format_arc(question, choice)
            if i == 0 and j == 0:
                print(prompt)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)

            label = 1 if labels[j] == answer_key else 0
            all_labels.append(label)

    return all_prompts, all_labels
  
  
def format_arc_end_q(question, choice, rand_question):
    return f"Q: {question} A: {choice} Q: {rand_question}"

def tokenized_arc_gen_end_q(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)):
        question = add_ending(dataset[i]['question'], '?')
        rand_idx = np.random.randint(len(dataset))
        rand_question = add_ending(dataset[rand_idx]['question'], '?')
        answer_key = dataset[i]['answerKey']
        answer_by_key = dict(zip(dataset[i]['choices']['label'], dataset[i]['choices']['text']))

        for key, answer in answer_by_key.items():
            answer = add_ending(answer, '.')
            prompt = format_arc_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)
            if key == answer_key:
                all_labels.append(1)
            else:
                all_labels.append(0)
            all_categories.append('')

    return all_prompts, all_labels, all_categories
