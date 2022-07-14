#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f
from ..utils.module_with_records_and_reducer import ModuleWithRecordsAndReducer

class BaseWeightRegularizer(ModuleWithRecordsAndReducer):
    def __init__(self, normalize_weights=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_weights = normalize_weights
        self.add_to_recordable_attributes(name="avg_weight_norm", is_stat=True)

    def compute_loss(self, weights):
        raise NotImplementedError

    def forward(self, weights):
        """
        weights should have shape (num_classes, embedding_size)
        """
        self.reset_stats()
        if self.normalize_weights:
            weights = torch.nn.functional.normalize(weights, p=2, dim=1)
        self.weight_norms = torch.norm(weights, p=2, dim=1)
        self.avg_weight_norm = torch.mean(self.weight_norms)
        loss_dict = self.compute_loss(weights)
        return self.reducer(loss_dict, weights, c_f.torch_arange_from_size(weights))