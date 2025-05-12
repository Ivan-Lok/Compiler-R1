#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT模型批量评估脚本 (多进程版):
针对指定目录下的多个LLVM IR数据集，使用不同的生成答案数量，
通过加载多个模型实例到不同进程/GPU并行处理文件，评估优化pass序列效果（OverOz），
计算平均值，并输出结果表格和折线图。
"""
import os
import sys
import json
import argparse
import torch
import glob
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import multiprocessing as mp # Use multiprocessing
from queue import Empty # For queue timeout
# import traceback # Uncomment for debugging

# --- Import Helper Functions ---
try:
    # Assume these helpers are safe to be imported in multiple processes
    from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_autophase import get_autophase_obs
    from agent_r1.tool.tools.comiler_autotuning.raw_tool.get_instrcount import get_instrcount
except ImportError:
    print("错误：无法导入 'get_autophase_obs' 或 'get_instrcount'。", file=sys.stderr)
    print("请确保 'agent_r1' 包在 PYTHONPATH 中，或者调整脚本中的导入路径。", file=sys.stderr)
    sys.exit(1)

# --- Helper Functions (Mostly Unchanged) ---

def parse_optimization_sequence(sequence_str: str):
    """Parse the optimization sequence string into a list.
    
    Args:
        sequence_str: Optimization sequence string from <answer> tag
        
    Returns:
        List of optimization options
    """
    if not sequence_str:
        return []
                
    # If JSON parsing fails, try to extract individual optimization passes
    passes = re.findall(r'--?[a-zA-Z0-9-]+', sequence_str)
    if passes:
        return passes
            
    return []

def get_overOz(ll_code, opt_flags, llvm_tools_path=None):
    """Calculates OverOz score."""
    if not isinstance(opt_flags, list) or not all(isinstance(f, str) for f in opt_flags):
        return None
    try:
        ic_value = get_instrcount(ll_code, *opt_flags, llvm_tools_path=llvm_tools_path)
        oz_value = get_instrcount(ll_code, [" "], llvm_tools_path=llvm_tools_path)
        if oz_value is None or ic_value is None: return None
        try:
            oz_value = float(oz_value)
            ic_value = float(ic_value)
        except (ValueError, TypeError): return None
        # if oz_value == 0: return None
        return (oz_value - ic_value) / oz_value
    except Exception:
        return None

def read_llvm_ir_file(file_path):
    """Reads LLVM IR code."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception:
        return None

def get_autophase_features(ll_code):
    """Gets autophase features."""
    if ll_code is None: return None
    try:
        features = get_autophase_obs(ll_code)
        return features if isinstance(features, dict) else None
    except Exception:
        return None

def parse_pass_sequence(response_text):
    """Parses pass sequence from model output string."""
    if not response_text: return []
    try:
        response_text = response_text.strip()
        if response_text.startswith("```python"): response_text = response_text[len("```python"):].strip()
        elif response_text.startswith("```"): response_text = response_text[3:].strip()
        if response_text.endswith("```"): response_text = response_text[:-3].strip()
        seq = eval(response_text)
        if isinstance(seq, list): return [str(p).strip() for p in seq if str(p).strip()]
    except Exception:
        passes = response_text.strip().split()
        return [p for p in passes if p]
    return []

import multiprocessing as mp
from queue import Empty
# import traceback

# ... (Helper 函数 get_overOz, read_llvm_ir_file, get_autophase_features, parse_pass_sequence 保持不变) ...

# --- Worker Process Function ---
def worker_process(
    worker_id,
    file_chunk,
    model_path,
    base_model,
    raw_tool_path,
    num_answers,
    max_length,
    max_retries,
    # **** CHANGE: Pass the RELATIVE device index ****
    relative_device_idx,
    results_queue,
    log_queue
):
    """
    Function executed by each worker process.
    Loads its own model and processes its assigned files using the relative device index.
    """
    # **** Use the relative index for device setup ****
    device_str = f"cuda:{relative_device_idx}"
    process_name = f"Worker-{worker_id}(RelGPU:{relative_device_idx})" # Log relative index

    log_queue.put(f"[{process_name}] Process started. Setting device to {device_str}")

    try:
        # --- Set the device for this process using the relative index ---
        # This is technically optional if device_map handles it, but good practice.
        torch.cuda.set_device(relative_device_idx)
        device = torch.device(device_str)

        log_queue.put(f"[{process_name}] Device set. Loading model onto {device}")

        # --- Load Model and Tokenizer *within* the process ---
        tokenizer = None
        model = None
        # ... (Model/Tokenizer loading logic - uses 'device' which is now correct) ...
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # log_queue.put(f"[{process_name}] Tokenizer loaded from {model_path}")
        except Exception:
            log_queue.put(f"[{process_name}] Tokenizer load failed from {model_path}, trying base {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            # log_queue.put(f"[{process_name}] Tokenizer loaded from {base_model}")

        chat_format = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None

        # Load model directly onto the assigned relative device
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # Pass the torch.device object directly
            device_map=device,
            trust_remote_code=True
        )
        model.eval()
        log_queue.put(f"[{process_name}] Model loaded successfully onto {device}.")

    except Exception as e:
        # Catch CUDA errors during setup too
        log_queue.put(f"[{process_name}] FATAL during setup on {device_str}: {e}. Exiting.")
        # traceback.print_exc(file=sys.stderr) # More detailed error for debugging
        for file_path in file_chunk:
            results_queue.put((file_path, None))
        return

    # --- Process Assigned Files ---
    # ... (File processing loop remains largely the same, using 'device') ...
    num_files_processed = 0
    for file_path in file_chunk:
        file_basename = os.path.basename(file_path)
        try:
            ll_code = read_llvm_ir_file(file_path)
            if ll_code is None:
                results_queue.put((file_path, None)); continue

            features = get_autophase_features(ll_code)
            if features is None:
                results_queue.put((file_path, None)); continue

            initial_inst_count = features.get('TotalInsts', 'N/A')
            try: initial_inst_count = int(initial_inst_count)
            except (ValueError, TypeError):
                results_queue.put((file_path, None)); continue

            instruction = f"""Act as a compiler optimization expert finding an optimal pass sequence for LLVM IR, aiming to reduce the total instruction count.
The LLVM IR code is represented by autophase features, the initial autophase features are:
```json
{json.dumps(features, indent=2)}
```
Initial instruction count: {initial_inst_count}
"""

            if chat_format:
                messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": instruction}]
                formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_input = f"<question>\n{instruction}\n</question>\n<answer>"

            # Inputs are moved to the correct relative device
            inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=4096).to(device)

            best_file_overoz = -float('inf')
            found_valid_answer_for_file = False

            for _ in range(num_answers):
                retry_count = 0
                generated_sequence = None
                while retry_count < max_retries:
                    try:
                        with torch.no_grad():
                            output = model.generate(
                                **inputs, max_new_tokens=max_length, do_sample=False,
                                temperature=0.7, top_p=0.9, top_k=50,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        full_generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

                        answer_start_tag = "<answer>"
                        answer_end_tag = "</answer>"
                        start_index = full_generated_text.rfind(answer_start_tag)
                        if start_index != -1:
                            content_start = start_index + len(answer_start_tag)
                            end_index = full_generated_text.find(answer_end_tag, content_start)
                            if end_index != -1:
                                response_text = full_generated_text[content_start:end_index].strip()
                                current_sequence = parse_optimization_sequence(response_text)
                                if current_sequence:
                                    generated_sequence = current_sequence
                                    break
                        retry_count += 1
                        # print(f"[{process_name}] Already retyr {retry_count} times.")
                    except Exception:
                        retry_count += 1
                        # time.sleep(0.2)

                if generated_sequence:
                    over_oz_value = get_overOz(ll_code, generated_sequence, raw_tool_path)
                    if over_oz_value is not None and isinstance(over_oz_value, (int, float)):
                        found_valid_answer_for_file = True
                        if over_oz_value > best_file_overoz:
                            best_file_overoz = over_oz_value

            results_queue.put((file_path, best_file_overoz if found_valid_answer_for_file else None))
            num_files_processed += 1

        except Exception as e:
            log_queue.put(f"[{process_name}] ERROR processing file {file_basename}: {e}")
            results_queue.put((file_path, None))

    log_queue.put(f"[{process_name}] Finished processing {num_files_processed}/{len(file_chunk)} assigned files.")


# ... (log_listener function remains the same) ...
def log_listener(log_queue):
    """Listens to the log queue and prints messages."""
    while True:
        try:
            record = log_queue.get()
            if record is None: # Sentinel value to stop
                break
            print(record, flush=True) # Print log messages from workers
        except (EOFError, KeyboardInterrupt):
            break # Exit if queue is closed or interrupted


# --- Core Evaluation Logic (Modified for Multiprocessing) ---
def evaluate_dataset_mp(
    model_path, base_model, ll_dir, raw_tool_path, num_answers,
    max_length, max_retries, num_workers,
    # **** CHANGE: No longer need visible_gpu_ids here, as they are implicitly handled ****
    # visible_gpu_ids
):
    """
    Evaluates a dataset using multiprocessing.
    Manages worker processes and collects results.
    """
    ll_files = glob.glob(os.path.join(ll_dir, '*.ll'))
    if not ll_files:
        print(f"  警告: 目录 {ll_dir} 中未找到 .ll 文件。", file=sys.stderr)
        return None

    dataset_name = os.path.basename(ll_dir)
    # **** Get the count of VISIBLE GPUs for the parent process ****
    num_visible_gpus = torch.cuda.device_count()
    if num_visible_gpus == 0:
         print("错误: 在主进程中未检测到可见的 CUDA 设备 (检查 CUDA_VISIBLE_DEVICES)。", file=sys.stderr)
         return None
    print(f"  开始处理数据集: {dataset_name} (文件数: {len(ll_files)}, 答案数: {num_answers}, 工作进程数: {num_workers}, 可见 GPUs: {num_visible_gpus})")


    start_time_dataset = time.time()
    file_results = {}

    manager = mp.Manager()
    results_queue = manager.Queue()
    log_queue = manager.Queue()

    log_thread = mp.Process(target=log_listener, args=(log_queue,), daemon=True)
    log_thread.start()

    files_per_worker = len(ll_files) // num_workers
    extra_files = len(ll_files) % num_workers
    file_chunks = []
    start_idx = 0
    for i in range(num_workers):
        end_idx = start_idx + files_per_worker + (1 if i < extra_files else 0)
        file_chunks.append(ll_files[start_idx:end_idx])
        start_idx = end_idx

    processes = []
    for i in range(num_workers):
        if not file_chunks[i]: continue
        # **** Calculate the RELATIVE device index for the worker ****
        # This assumes the child process sees visible GPUs indexed 0, 1, ...
        relative_device_idx = i % num_visible_gpus
        p = mp.Process(
            target=worker_process,
            args=(
                i, file_chunks[i], model_path, base_model, raw_tool_path,
                num_answers, max_length, max_retries,
                # **** Pass the RELATIVE index ****
                relative_device_idx,
                results_queue, log_queue
            ),
            daemon=True
        )
        processes.append(p)
        p.start()
        # Log the *relative* index the worker will use
        log_queue.put(f"[Main] Launched Worker-{i} for {len(file_chunks[i])} files on relative GPU index {relative_device_idx}")

    # --- Collect Results (Logic remains the same) ---
    total_files_expected = len(ll_files)
    results_received = 0
    progress_interval = max(1, total_files_expected // 20)
    start_collect_time = time.time() # Track collection time separately

    while results_received < total_files_expected:
        try:
            # Increase timeout slightly?
            file_path, best_overoz = results_queue.get(timeout=300) # 5 min timeout
            file_results[file_path] = best_overoz
            results_received += 1
            if results_received % progress_interval == 0 or results_received == total_files_expected:
                elapsed_time = time.time() - start_collect_time
                print(f"    进度 ({dataset_name}, Ans={num_answers}): {results_received}/{total_files_expected} 文件结果已接收 ({elapsed_time:.1f} 秒)...", end='\r')

        except Empty:
            # Check if workers are alive *before* declaring timeout failure
            any_worker_alive = any(p.is_alive() for p in processes)
            current_time = time.time()
            if any_worker_alive and (current_time - start_collect_time) < 900: # 15 min overall timeout?
                # If workers are alive and total time isn't excessive, keep waiting briefly
                log_queue.put(f"[Main] Queue empty, workers still alive ({sum(p.is_alive() for p in processes)}/{len(processes)}). Waiting...")
                time.sleep(10) # Wait a bit longer before next check
                continue
            elif not any_worker_alive:
                 log_queue.put("[Main] ERROR: Queue empty and all workers have exited prematurely. Aborting dataset.")
            else: # Workers alive, but exceeded overall timeout
                 log_queue.put(f"[Main] ERROR: Timeout ({current_time - start_collect_time:.0f}s) waiting for results. Workers alive: {sum(p.is_alive() for p in processes)}/{len(processes)}. Aborting dataset.")

            # Mark remaining files as failed if aborting
            for f in ll_files:
                if f not in file_results: file_results[f] = None
            results_received = total_files_expected # Force loop exit
            break
        except (KeyboardInterrupt, SystemExit):
             log_queue.put("[Main] Interrupted. Terminating workers...")
             for p in processes:
                 if p.is_alive(): p.terminate()
             break
        except Exception as e:
             log_queue.put(f"[Main] ERROR collecting results: {e}")
             break

    print() # Newline after progress

    # --- Wait for Workers to Finish ---
    log_queue.put("[Main] Result collection finished or aborted. Joining workers...")
    for i, p in enumerate(processes):
        try:
            p.join(timeout=30)
            if p.is_alive():
                log_queue.put(f"[Main] Worker-{i} did not exit cleanly, terminating.")
                p.terminate()
                p.join()
        except Exception as e:
             log_queue.put(f"[Main] Error joining worker {i}: {e}")

    log_queue.put(None)
    log_thread.join(timeout=5)

    # --- Calculate Average (Logic remains the same) ---
    valid_scores = [score for score in file_results.values() if score is not None and isinstance(score, (int, float))]
    end_time_dataset = time.time()
    total_files_submitted = len(ll_files)
    files_with_valid_results = len(valid_scores)

    print(f"  完成数据集: {dataset_name}. 提交文件: {total_files_submitted}. 有效结果文件: {files_with_valid_results}. 耗时: {end_time_dataset - start_time_dataset:.2f} 秒.")

    if valid_scores:
        avg_overoz = np.mean(valid_scores)
        print(f"  {dataset_name} (Ans={num_answers}) 平均最佳 OverOz: {avg_overoz:.4f}")
        return avg_overoz
    else:
        print(f"  {dataset_name} (Ans={num_answers}): 未能计算有效的 OverOz 分数。")
        return None

# --- Main Batch Evaluation Logic ---
def main():
    # ... (Argument parsing remains the same) ...
    parser = argparse.ArgumentParser(description="批量评估SFT模型在多个数据集和答案数量下的性能 (多进程)")
    # Model & Paths
    parser.add_argument("--model_path", type=str, required=True, help="训练好的SFT模型检查点路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen1.5-1.5B-Chat", help="基础模型名称 (用于tokenizer备用)")
    parser.add_argument("--test_base_dir", type=str, required=True, help="包含所有测试数据集子目录的基础路径")
    parser.add_argument("--raw_tool_path", type=str, required=True, help="传递给 get_overOz 的原始工具路径")
    # Evaluation Parameters
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="要测试的数据集子目录名称列表")
    parser.add_argument("--num_answers_list", type=int, nargs='+', required=True, help="要测试的每个文件生成的答案数量列表")
    # Generation Parameters
    parser.add_argument("--max_length", type=int, default=350, help="模型生成的最大新 token 数量")
    parser.add_argument("--max_retries", type=int, default=20, help="每个答案的生成重试次数")
    # Hardware & Parallelism
    parser.add_argument("--num_workers", type=int, default=4, help="要启动的并行工作进程数")
    # Output
    parser.add_argument("--output_table", type=str, default="evaluation_results_mp.txt", help="保存结果表格的文件名")
    parser.add_argument("--output_plot", type=str, default="evaluation_plot_mp.png", help="保存结果折线图的文件名")
    args = parser.parse_args()

    # --- Setup ---
    if args.num_workers <= 0:
        print("错误: --num_workers 必须是正整数。", file=sys.stderr)
        sys.exit(1)

    # **** No need to parse visible_gpu_ids here, evaluate_dataset_mp will check torch.cuda.device_count() ****
    if not torch.cuda.is_available():
        print("错误: 未找到或配置CUDA设备。", file=sys.stderr)
        sys.exit(1)

    print(f"测试数据集: {', '.join(args.datasets)}")
    print(f"测试答案数: {', '.join(map(str, args.num_answers_list))}")
    print(f"模型路径: {args.model_path}")
    print(f"工作进程数: {args.num_workers}")
    # Let user know which devices *should* be visible based on environment
    print(f"环境变量 CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set (all visible)')}'")

    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Warning: Could not set start method to 'spawn'. Using default.")

    # --- Run Evaluations ---
    results = {}
    overall_start_time = time.time()
    sorted_num_answers = sorted(args.num_answers_list)
    sorted_datasets = sorted(args.datasets)

    for num_ans in sorted_num_answers:
        print(f"\n===== 开始评估 Num Answers = {num_ans} =====")
        results[num_ans] = {}
        for dataset_name in sorted_datasets:
            dataset_dir = os.path.join(args.test_base_dir, dataset_name)
            if not os.path.isdir(dataset_dir):
                 print(f"警告: 数据集目录 {dataset_dir} 不存在，跳过。", file=sys.stderr)
                 results[num_ans][dataset_name] = np.nan
                 continue

            avg_overoz = evaluate_dataset_mp( # Call the updated function
                args.model_path, args.base_model, dataset_dir, args.raw_tool_path,
                num_ans, args.max_length, args.max_retries,
                args.num_workers
                # Removed visible_gpu_ids argument here
            )
            results[num_ans][dataset_name] = avg_overoz if avg_overoz is not None else np.nan

    overall_end_time = time.time()
    print(f"\n===== 所有评估完成. 总耗时: {(overall_end_time - overall_start_time) / 60:.2f} 分钟 =====")

    # --- Generate Output Table and Plot (Identical Logic) ---
    # ... (Table and Plot generation remains the same) ...
    print(f"\n生成结果表格到: {args.output_table}")
    header = f"{'Num Answers':<15}" + "".join([f"{name:>15}" for name in sorted_datasets])
    table_lines = [header, "-" * len(header)]
    for num_ans in sorted_num_answers:
        row = f"{num_ans:<15}"
        for dataset_name in sorted_datasets:
             score = results[num_ans].get(dataset_name, np.nan)
             row += f"{score:>15.4f}" if not np.isnan(score) else f"{'N/A':>15}"
        table_lines.append(row)
    try:
        with open(args.output_table, 'w') as f:
            f.write("\n".join(table_lines))
        print("结果表格写入成功。")
    except IOError as e:
        print(f"错误: 无法写入结果表格文件 {args.output_table}: {e}", file=sys.stderr)

    print(f"\n生成结果折线图到: {args.output_plot}")
    try:
        plt.figure(figsize=(12, 7))
        for dataset_name in sorted_datasets:
            y_values = [results[num_ans].get(dataset_name, np.nan) for num_ans in sorted_num_answers]
            if not all(np.isnan(y_values)):
                 plt.plot(sorted_num_answers, y_values, marker='o', linestyle='-', label=dataset_name)
            else:
                 print(f"  跳过绘制数据集 '{dataset_name}' (无有效数据点)")
        plt.xlabel("Number of Generated Answers per File")
        plt.ylabel("Average Best OverOz Score")
        plt.title(f"Model Performance vs. Number of Answers\nModel: {os.path.basename(args.model_path)}")
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(sorted_num_answers)
        all_valid_scores = [score for num_ans_dict in results.values() for score in num_ans_dict.values() if not np.isnan(score)]
        if all_valid_scores:
             min_score = min(all_valid_scores)
             max_score = max(all_valid_scores)
             plt.ylim(min_score - 0.05 * abs(min_score) - 0.01, max_score + 0.05 * abs(max_score) + 0.01)

        plt.tight_layout()
        plt.savefig(args.output_plot)
        plt.close()
        print("结果折线图保存成功。")
    except Exception as e:
        print(f"错误: 无法生成或保存绘图文件 {args.output_plot}: {e}", file=sys.stderr)
        print("请确保已安装 matplotlib: pip install matplotlib", file=sys.stderr)


if __name__ == "__main__":
    mp.freeze_support()
    main()