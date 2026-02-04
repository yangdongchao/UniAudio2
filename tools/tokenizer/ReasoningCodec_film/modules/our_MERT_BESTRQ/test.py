import torch
from dataclasses import dataclass
from logging import getLogger
import torch.nn.functional as F
import fairseq

logger = getLogger(__name__)

@dataclass
class UserDirModule:
    user_dir: str

def load_best_rq_model(model_dir, checkpoint_dir):
    '''Load Fairseq SSL model'''
    #导入模型所在的代码模块
    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    
    #载入模型的checkpoint
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir], strict=False)
    model = model[0]

    return model

if __name__ == '__main__':

    pass
