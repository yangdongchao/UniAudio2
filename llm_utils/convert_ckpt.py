from safetensors.torch import load_file
import torch

def convert_to_pt(model_dir):
    # 加载索引文件
    import json
    with open(f"{model_dir}/model.safetensors.index.json") as f:
        index = json.load(f)
    
    # 合并所有分片
    state_dict = {}
    for weight_file in index["weight_map"].values():
        state_dict.update(load_file(f"{model_dir}/{weight_file}"))
    
    # 保存为 .pt 文件
    torch.save(state_dict)
    