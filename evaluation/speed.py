import json
import argparse
from transformers import AutoTokenizer
import numpy as np


def speed(jsonl_file, jsonl_file_base, tokenizer, task=None, report=True):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer)
    mt_bench_list = ["writing", "roleplay", "reasoning", "math" , "coding", "extraction", "stem", "humanities"]

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if task=="overall":
                data.append(json_obj)
            elif task == "mt_bench":
                if json_obj["category"] in mt_bench_list:
                    data.append(json_obj)
            else:
                if json_obj["category"] == task:
                    data.append(json_obj)

    speeds=[]
    accept_lengths_list = []
    for datapoint in data:
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
        speeds.append(tokens/times)

    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if task=="overall":
                data.append(json_obj)
            elif task == "mt_bench":
                if json_obj["category"] in mt_bench_list:
                    data.append(json_obj)
            else:
                if json_obj["category"] == task:
                    data.append(json_obj)

    total_time=0
    total_token=0
    speeds0=[]
    for datapoint in data:
        answer=datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens / times)
        total_time+=times
        total_token+=tokens

    tokens_per_second = np.array(speeds).mean()
    tokens_per_second_baseline = np.array(speeds0).mean()
    speedup_ratio = np.array(speeds).mean() / np.array(speeds0).mean()

    if report:
        print("="*30, "Task: ", task, "="*30)
        print("#Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('Tokens per second: ', tokens_per_second)
        print('Tokens per second for the baseline: ', tokens_per_second_baseline)
        print("Speedup ratio: ", speedup_ratio)
    return tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths_list


def get_single_speedup(jsonl_file, jsonl_file_base):
    tokenizer_path="/cache/CKPT/vicuna-7b-v1.3/"
    for subtask_name in ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]:
        speed(jsonl_file, jsonl_file_base, tokenizer_path, task=subtask_name)


def get_mean_speedup():
    tokenizer_path="/cache/CKPT/vicuna-7b-v1.3/"
    jsonl_file_name = "vicuna-7b-v1.3-lade-level-5-win-7-guess-7-float16.jsonl"
    jsonl_file_base_name = "vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl"
    jsonl_file_run_list = [
        "../data/spec_bench/model_answer_temp0_run_1/{}".format(jsonl_file_name),
        "../data/spec_bench/model_answer_temp0_run_2/{}".format(jsonl_file_name),
        "../data/spec_bench/model_answer_temp0_run_3/{}".format(jsonl_file_name)
                           ]
    jsonl_file_base_run_list = [
        "../data/spec_bench/model_answer_temp0_run_1/{}".format(jsonl_file_base_name),
        "../data/spec_bench/model_answer_temp0_run_2/{}".format(jsonl_file_base_name),
        "../data/spec_bench/model_answer_temp0_run_3/{}".format(jsonl_file_base_name)
                           ]

    for subtask_name in ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]:
        print("=" * 30, "Task: ", subtask_name, "=" * 30)
        tokens_per_second_list = []
        tokens_per_second_baseline_list = []
        speedup_ratio_list = []
        accept_lengths_list = []
        for jsonl_file, jsonl_file_base in zip(jsonl_file_run_list, jsonl_file_base_run_list):
            tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths = speed(jsonl_file, jsonl_file_base, tokenizer_path, task=subtask_name, report=False)
            tokens_per_second_list.append(tokens_per_second)
            tokens_per_second_baseline_list.append(tokens_per_second_baseline)
            speedup_ratio_list.append(speedup_ratio)
            accept_lengths_list.extend(accept_lengths)

        avg_accept_lengths = np.mean(accept_lengths_list)
        print("#Mean accepted tokens: {}".format(avg_accept_lengths))

        avg = np.mean(tokens_per_second_list)
        std = np.std(tokens_per_second_list, ddof=1)  # np.sqrt(( a.var() * a.size) / (a.size - 1))
        print("Tokens per second: Mean result: {}, Std result: {}".format(avg, std))

        avg_baseline = np.mean(tokens_per_second_baseline_list)
        std_baseline = np.std(tokens_per_second_baseline_list, ddof=1)  # np.sqrt(( a.var() * a.size) / (a.size - 1))
        print("Tokens per second (baseline): Mean result: {}, Std result: {}".format(avg_baseline, std_baseline))

        avg_speedup = np.mean(speedup_ratio_list)
        std_speedup = np.std(speedup_ratio_list, ddof=1)  # np.sqrt(( a.var() * a.size) / (a.size - 1))
        print("Speedup ratio: Mean result: {}, Std result: {}".format(avg_speedup, std_speedup))
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-path",
        default='../data/spec_bench/model_answer/vicuna-7b-v1.3-sps-68m-float16-temp-0.0.jsonl',
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--base-path",
        default='../data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl',
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--mean-report",
        action="store_true",
        default=False,
        help="report mean speedup over different runs")

    args = parser.parse_args()
    if args.mean_report:
        get_mean_speedup()
    else:
        get_single_speedup(jsonl_file=args.file_path, jsonl_file_base=args.base_path)