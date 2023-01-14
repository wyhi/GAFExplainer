import torch
import torch.nn as nn
from typing import List, Union
from torch import Tensor
import numpy as np
from torch_geometric.data.data import Data
from torch_geometric.nn import MessagePassing


def fidelity(ori_probs: torch.Tensor, unimportant_probs: torch.Tensor) -> float:
    drop_probability = ori_probs - unimportant_probs
    return drop_probability.mean().item()


def fidelity_inv(ori_probs: torch.Tensor, important_probs: torch.Tensor) -> float:
    drop_probability = ori_probs - important_probs
    return drop_probability.mean().item()


class XCollector:
    """
    数据收集器，用来计算fidelity+,fidelity-
    """

    def __init__(self, sparsity=None):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': [],
                                                'sparsity': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []
        self.__sparsity = sparsity
        self.__fidelity, self.__fidelity_inv = None, None
        self.__score = None

    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': [],
                                                'sparsity': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []
        self.__fidelity, self.__fidelity_inv = None, None

    def collect_data(self, masks: List[Tensor],
                     related_preds: dir, label: int = 0) -> None:
        if self.__fidelity or self.__fidelity_inv:
            self.__fidelity, self.__fidelity_inv = None, None
            print(f'#w#Called collect_data() after calculate explainable metrics.')

        if not np.isnan(label):  # 其实这里有点没理解，related_preds列表里只有一个数据，因此==0就行了
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)
            for key in self.__related_preds.keys():
                if key not in related_preds[label].keys():
                    self.__related_preds[key].append(None)
            self.__targets.append(label)
            self.masks.append(masks)
    @property
    def fidelity(self):
        if self.__fidelity:
            return self.__fidelity
        elif None in self.__related_preds['maskout'] or None in self.__related_preds['origin']:
            return None
        else:
            mask_out_preds = torch.tensor(self.__related_preds['maskout'])
            one_mask_preds = torch.tensor(self.__related_preds['origin'])
            self.__fidelity = fidelity(one_mask_preds, mask_out_preds)
            return self.__fidelity

    @property
    def fidelity_inv(self):
        if self.__fidelity_inv:
            return self.__fidelity_inv
        elif None in self.__related_preds['masked'] or None in self.__related_preds['origin']:
            return None
        else:
            masked_preds=torch.tensor(self.__related_preds['masked'])
            one_mask_preds=torch.tensor(self.__related_preds['origin'])
            self.__fidelity_inv=fidelity_inv(one_mask_preds,masked_preds)
            return self.__fidelity_inv

    @property
    def sparsity(self):
        if self.__sparsity:
            return self.__sparsity
        elif None in self.__related_preds['sparsity']:
            return None
        else:
            return torch.tensor(self.__related_preds['sparsity']).mean().item()

