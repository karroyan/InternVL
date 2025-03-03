import argparse
import torch
from safetensors import safe_open
from internvl.model.internvl_chat import InternVLChatModel
from internvl.model.internvl_chat.modeling_internvl_classification import InternVLSequenceClassificationModel
from transformers import AutoTokenizer

def merge_weights_and_log(model, saved_weights, log_file):
    """
    合并权重并记录合并结果
    :param model: 模型对象
    :param saved_weights: 加载的权重字典
    :param log_file: 日志文件路径
    """
    with open(log_file, 'w') as f:
        # 合并 classify_attention 权重
        if 'classify_attention.weight' in saved_weights and 'classify_attention.bias' in saved_weights:
            model.classify_attention.weight.data = saved_weights['classify_attention.weight']
            model.classify_attention.bias.data = saved_weights['classify_attention.bias']
            f.write("'classify_attention.weight' 和 'classify_attention.bias' 已 merge\n")
        else:
            f.write("'classify_attention.weight' 和 'classify_attention.bias' 未 merge（缺少权重）\n")

        # 合并 classifier_heads 权重
        if 'classifier_heads.weight' in saved_weights:
            model.classifier_heads.weight.data = saved_weights['classifier_heads.weight']
            f.write("'classifier_heads.weight' 已 merge\n")
        else:
            f.write("'classifier_heads.weight' 未 merge（缺少权重）\n")

        # 检查其他权重是否被 merge
        for key in saved_weights.keys():
            if key not in [
                'classify_attention.weight',
                'classify_attention.bias',
                'classifier_heads.weight'
            ]:
                f.write(f"'{key}' 未 merge（非目标权重）\n")

        
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to the input model')
    parser.add_argument('--output_path', type=str, help='Path to the output model')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint directory')
    parser.add_argument('--log_file', type=str, default='merge_log.txt', help='Path to the log file')
    args = parser.parse_args()

    # 加载模型
    print('Loading model...')
    model = InternVLSequenceClassificationModel.from_pretrained(
        args.input_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()

    # 加载 tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)

    # 加载保存的权重
    print('Loading saved weights...')
    saved_weights = {}
    with safe_open(f'{args.checkpoint_path}/model-00001-of-00002.safetensors', framework="pt") as f:
        saved_weights.update({k: f.get_tensor(k) for k in f.keys()})
    with safe_open(f'{args.checkpoint_path}/model-00002-of-00002.safetensors', framework="pt") as f:
        saved_weights.update({k: f.get_tensor(k) for k in f.keys()})

    # 合并权重并记录日志
    print('Merging weights...')
    merge_weights_and_log(model, saved_weights, args.log_file)

    # 处理 LoRA 部分
    if model.config.use_backbone_lora:
        model.vision_model.merge_and_unload()
        model.vision_model = model.vision_model.model
        model.config.use_backbone_lora = 0
    if model.config.use_llm_lora:
        model.language_model.merge_and_unload()
        model.language_model = model.language_model.model
        model.config.use_llm_lora = 0

    # 检查模型中是否存在 meta 张量
    for name, param in model.named_parameters():
        if param.is_meta:
            print(f"警告: '{name}' 是 meta 张量，未正确加载数据\n")
            # 手动初始化 meta 张量
            param.data = torch.zeros_like(param.data, device=param.device)
            print(f"已初始化 '{name}' 为全零张量\n")


    # 保存模型
    print('Saving model...')
    model.save_pretrained(args.output_path)

    # 保存 tokenizer
    print('Saving tokenizer...')
    tokenizer.save_pretrained(args.output_path)

    print('Done!')

if __name__ == '__main__':
    main()