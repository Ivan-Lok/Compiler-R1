#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单的SFT模型测试脚本
不依赖于Ray和vllm，直接使用transformers库加载模型并进行推理
"""

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="测试SFT模型")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--use_gpu", 
        action="store_true", 
        default=True,
        help="是否使用GPU"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=30000,
        help="最大生成长度"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="基础模型名称，用于获取正确的聊天模板"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 如果未指定模型路径，尝试查找最新的检查点
    if args.model_path is None:
        sft_model_path = os.path.expanduser("~/outputs/compiler_autotuning_sft")
        checkpoints = sorted([d for d in os.listdir(sft_model_path) if d.startswith("global_step_")])
        
        if not checkpoints:
            print(f"错误: 在 {sft_model_path} 中未找到检查点")
            return
        
        args.model_path = os.path.join(sft_model_path, checkpoints[-1])
    
    print(f"使用模型路径: {args.model_path}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"使用设备: {device}")
    
    # 创建测试输入
    autophase_features = {
        "BBNumArgsHi": 0,
        "BBNumArgsLo": 10,
        "onePred": 134,
        "onePredOneSuc": 93,
        "onePredTwoSuc": 28,
        "oneSuccessor": 112,
        "twoPred": 53,
        "twoPredOneSuc": 15,
        "twoEach": 36,
        "twoSuccessor": 70,
        "morePreds": 3,
        "BB03Phi": 10,
        "BBHiPhi": 0,
        "BBNoPhi": 191,
        "BeginPhi": 10,
        "BranchCount": 182,
        "returnInt": 39,
        "CriticalCount": 22,
        "NumEdges": 252,
        "const32Bit": 182,
        "const64Bit": 158,
        "numConstZeroes": 95,
        "numConstOnes": 143,
        "UncondBranches": 112,
        "binaryConstArg": 61,
        "NumAShrInst": 0,
        "NumAddInst": 38,
        "NumAllocaInst": 68,
        "NumAndInst": 0,
        "BlockMid": 21,
        "BlockLow": 180,
        "NumBitCastInst": 82,
        "NumBrInst": 182,
        "NumCallInst": 152,
        "NumGetElementPtrInst": 78,
        "NumICmpInst": 70,
        "NumLShrInst": 0,
        "NumLoadInst": 338,
        "NumMulInst": 2,
        "NumOrInst": 0,
        "NumPHIInst": 10,
        "NumRetInst": 11,
        "NumSExtInst": 50,
        "NumSelectInst": 1,
        "NumShlInst": 0,
        "NumStoreInst": 179,
        "NumSubInst": 47,
        "NumTruncInst": 5,
        "NumXorInst": 0,
        "NumZExtInst": 27,
        "TotalBlocks": 201,
        "TotalInsts": 1348,
        "TotalMemInst": 815,
        "TotalFuncs": 31,
        "ArgsPhi": 20,
        "testUnary": 570
    }
    
    # 使用聊天格式创建输入
    instruction = f"""

Act as a compiler optimization expert finding an optimal pass sequence for LLVM IR, aiming to reduce the total instruction count.

The LLVM IR code is represented by autophase features, the initial autophase features are:
{json.dumps(autophase_features, indent=2)}
Initial instruction count: 1348

Your task is to:

Evaluate the provided Initial Candidate Pass Sequence using the instrcount tool to determine its instruction count improvement compared to the default -Oz optimization.
If the initial sequence provides a positive improvement (improvement_over_oz > 0), recommend it as the final answer.
If the initial sequence does not provide a positive improvement (improvement_over_oz <= 0), use the find_best_pass_sequence tool to search for a better sequence.
If the search finds a sequence with positive improvement (improvement_percentage > 0), recommend that sequence.
If the search tool fails to find a sequence with positive improvement, recommend the default ['-Oz'] sequence as the safest option.
Present your reasoning step-by-step using <think> tags and tool interactions using <tool_call> and <tool_response> structure, concluding with the final recommended sequence in an <answer> tag.
"""
    
    try:
        # 加载分词器 - 尝试从原始模型加载以确保聊天模板正确
        print("加载分词器...")
        try:
            # 首先尝试从SFT模型加载
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        except Exception as e:
            # 如果失败，从基础模型加载
            print(f"从SFT模型加载tokenizer失败: {e}")
            print(f"尝试从基础模型 {args.base_model} 加载")
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
        # 检查是否有聊天模板
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            print(f"检测到聊天模板: {tokenizer.chat_template[:50]}...")
            chat_format = True
        else:
            print("未检测到聊天模板，使用普通文本输入")
            chat_format = False
        
        # 准备输入
        if chat_format:
            # 使用聊天格式
            messages = [
                {"role": "system", "content": "You are a helpful assistant for compiler optimization."},
                {"role": "user", "content": instruction}
            ]
            
            # 使用tokenizer的聊天模板
            formatted_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"\n使用聊天模板格式化输入:\n{formatted_input}...")
        else:
            # 使用普通文本输入
            formatted_input = f"<question>\n{instruction}\n</question>\n<answer>"
            print(f"\n使用普通文本格式化输入:\n{formatted_input[:200]}...")
        
        print("\n加载模型...")
        try:
            # 如果路径是目录，使用AutoModelForCausalLM.from_pretrained
            if os.path.isdir(args.model_path):
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
            # 如果是文件路径，尝试直接加载模型状态
            elif os.path.isfile(args.model_path):
                print(f"检测到模型是文件路径，尝试加载状态字典...")
                # 首先加载基础模型
                model = AutoModelForCausalLM.from_pretrained(
                    args.base_model,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                # 加载状态字典
                model.load_state_dict(torch.load(args.model_path, map_location=device))
            else:
                raise ValueError(f"无效的模型路径: {args.model_path}")
        except Exception as e:
            print(f"加载模型时发生错误: {e}")
            print("尝试添加trust_remote_code=True...")
            # 再次尝试加载，添加信任远程代码选项
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        # 为了节省内存，我们使用半精度
        if torch.cuda.is_available():
            model.half()  # 使用FP16
        
        print("模型加载完成，开始生成...")
        
        # 编码输入
        inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成输出
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 提取模型回答（输入之后的内容）
        if chat_format:
            # 在聊天格式中，回答是整个生成文本减去输入
            assistant_prefix = "<|im_start|>assistant\n"
            if assistant_prefix in generated_text:
                response = generated_text.split(assistant_prefix)[-1].strip()
            else:
                response = generated_text[len(formatted_input):].strip()
        else:
            # 在普通文本中，回答是<answer>之后的文本
            parts = generated_text.split("<answer>")
            response = parts[1].strip() if len(parts) > 1 else generated_text
        
        print("\n" + "="*80)
        print("模型回答:")
        print("-"*80)
        print(response)
        print("="*80)
        
        # 保存结果到文件
        with open("model_response.txt", "w") as f:
            f.write(f"模型路径: {args.model_path}\n\n")
            f.write(f"输入:\n{formatted_input}\n\n")
            f.write(f"完整输出:\n{generated_text}\n\n")
            f.write(f"提取的回答:\n{response}")
        print(f"结果已保存到 model_response.txt")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 