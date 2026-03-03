"""
示例1: 数据集加载和格式化
演示如何使用RLTrainingTool加载和查看GSM8K数据集
"""

import sys
from pathlib import Path
import json

# 添加项目路径
project_root = Path(__file__).parent.parent / "HelloAgents"
sys.path.insert(0, str(project_root))

from hello_agents.tools import RLTrainingTool


# ============================================================================
# 示例1: 加载SFT格式数据集
# ============================================================================

def load_sft_dataset():
    """
    使用RLTrainingTool加载SFT格式的GSM8K数据集

    SFT数据格式:
    {
        "prompt": "Question: ...\n\nLet's solve this step by step:\n",
        "completion": "Step 1: ...\nFinal Answer: 42",
        "text": "Question: ...\n\nLet's solve this step by step:\nStep 1: ...\nFinal Answer: 42"
    }
    """
    tool = RLTrainingTool()

    config = {
        "action": "load_dataset",
        "format": "sft",
        "split": "train",
        "max_samples": 5
    }

    print("加载SFT格式数据集...")
    result = tool.run(config)
    result_dict = json.loads(result)

    print(f"✅ 数据集大小: {result_dict['dataset_size']}")
    print(f"📋 数据集列: {result_dict['sample_keys']}")
    print(f"\n💡 提示: 数据集已加载,可以用于训练")
    print(f"   使用 action='train' 开始训练")

    return result_dict


# ============================================================================
# 示例2: 加载RL格式数据集
# ============================================================================

def load_rl_dataset():
    """
    使用RLTrainingTool加载RL格式的GSM8K数据集

    RL数据格式:
    {
        "prompt": "<|im_start|>user\nQuestion: ...\n<|im_end|>\n<|im_start|>assistant\n",
        "ground_truth": "42",
        "question": "...",
        "full_answer": "..."
    }
    """
    tool = RLTrainingTool()

    config = {
        "action": "load_dataset",
        "format": "rl",
        "split": "train",
        "max_samples": 5,
        "model_name": "Qwen/Qwen3-0.6B"
    }

    print("加载RL格式数据集...")
    result = tool.run(config)
    result_dict = json.loads(result)

    print(f"✅ 数据集大小: {result_dict['dataset_size']}")
    print(f"📋 数据集列: {result_dict['sample_keys']}")
    print(f"\n💡 提示: RL数据集已加载,包含prompt和ground_truth")
    print(f"   可用于GRPO训练")

    return result_dict


# ============================================================================
# 示例3: 加载不同split的数据集
# ============================================================================

def load_different_splits():
    """
    加载训练集和测试集
    """
    tool = RLTrainingTool()
    
    # 加载训练集
    train_config = {
        "action": "load_dataset",
        "format": "sft",
        "split": "train",
        "max_samples": 100
    }
    
    print("加载训练集...")
    train_result = tool.run(train_config)
    train_data = json.loads(train_result)
    print(f"✅ 训练集: {train_data['dataset_size']} 样本")
    
    # 加载测试集
    test_config = {
        "action": "load_dataset",
        "format": "sft",
        "split": "test",
        "max_samples": 50
    }
    
    print("\n加载测试集...")
    test_result = tool.run(test_config)
    test_data = json.loads(test_result)
    print(f"✅ 测试集: {test_data['dataset_size']} 样本")
    
    return train_data, test_data


# ============================================================================
# 示例4: 加载完整数据集
# ============================================================================

def load_full_dataset():
    """
    加载完整数据集 (max_samples=None)
    
    GSM8K数据集:
    - 训练集: ~7500 样本
    - 测试集: ~1300 样本
    """
    tool = RLTrainingTool()
    
    config = {
        "action": "load_dataset",
        "format": "sft",
        "split": "train",
        "max_samples": None  # None = 使用全部数据
    }
    
    print("加载完整训练集...")
    print("⚠️  这可能需要一些时间...")
    
    # 实际加载时取消注释
    # result = tool.run(config)
    # result_dict = json.loads(result)
    # print(f"✅ 完整训练集: {result_dict['dataset_size']} 样本")
    
    print("💡 提示: 设置 max_samples=None 可以加载全部数据")
    print("   GSM8K训练集约有 7500 个样本")
    
    return config


# ============================================================================
# 示例5: 对比SFT和RL格式
# ============================================================================

def compare_sft_rl_formats():
    """
    对比SFT和RL数据格式的区别
    """
    tool = RLTrainingTool()

    print("="*80)
    print("SFT vs RL 数据格式对比")
    print("="*80)

    # SFT格式
    sft_config = {
        "action": "load_dataset",
        "format": "sft",
        "split": "train",
        "max_samples": 1
    }

    print("\n1. SFT格式:")
    sft_result = tool.run(sft_config)
    sft_data = json.loads(sft_result)
    print(f"   列: {sft_data['sample_keys']}")
    print(f"   用途: 监督微调 (Supervised Fine-Tuning)")
    print(f"   特点: 包含完整的prompt和completion")

    # RL格式
    rl_config = {
        "action": "load_dataset",
        "format": "rl",
        "split": "train",
        "max_samples": 1,
        "model_name": "Qwen/Qwen3-0.6B"
    }

    print("\n2. RL格式:")
    rl_result = tool.run(rl_config)
    rl_data = json.loads(rl_result)
    print(f"   列: {rl_data['sample_keys']}")
    print(f"   用途: 强化学习训练 (Reinforcement Learning)")
    print(f"   特点: 包含prompt和ground_truth,用于奖励计算")

    print("\n主要区别:")
    print("  - SFT: 直接学习正确答案")
    print("  - RL: 通过奖励信号学习,更灵活")

    return sft_data, rl_data


# ============================================================================
# 示例6: 数据集统计信息
# ============================================================================

def dataset_statistics():
    """
    查看数据集的统计信息
    """
    tool = RLTrainingTool()

    config = {
        "action": "load_dataset",
        "format": "sft",
        "split": "train",
        "max_samples": 100
    }

    print("加载数据集...")
    result = tool.run(config)
    result_dict = json.loads(result)

    print("\n数据集统计:")
    print(f"  总样本数: {result_dict['dataset_size']}")
    print(f"  数据列: {', '.join(result_dict['sample_keys'])}")
    print(f"  数据集: GSM8K (Grade School Math 8K)")
    print(f"  任务类型: 数学推理")

    print(f"\n💡 提示: 数据集包含以下字段:")
    for key in result_dict['sample_keys']:
        print(f"  - {key}")

    return result_dict


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("示例1: 加载SFT格式数据集")
    print("="*80)
    load_sft_dataset()
    
    print("\n" + "="*80)
    print("示例2: 加载RL格式数据集")
    print("="*80)
    load_rl_dataset()
    
    print("\n" + "="*80)
    print("示例3: 加载不同split的数据集")
    print("="*80)
    load_different_splits()
    
    print("\n" + "="*80)
    print("示例4: 加载完整数据集")
    print("="*80)
    load_full_dataset()
    
    print("\n" + "="*80)
    print("示例5: 对比SFT和RL格式")
    print("="*80)
    compare_sft_rl_formats()
    
    print("\n" + "="*80)
    print("示例6: 数据集统计信息")
    print("="*80)
    dataset_statistics()

