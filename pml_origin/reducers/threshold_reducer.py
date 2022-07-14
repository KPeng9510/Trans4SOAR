from .base_reducer import BaseReducer
import torch


class ThresholdReducer(BaseReducer):
    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_to_recordable_attributes(name="threshold")

    def element_reduction(self, losses, *_):
        return self.element_reduction_helper(losses, "elements_above_threshold")
    
    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction_helper(losses, "pos_pairs_above_threshold")

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction_helper(losses, "neg_pairs_above_threshold")

    def triplet_reduction(self, losses, *args):
        return self.element_reduction_helper(losses, "triplets_above_threshold")

    def element_reduction_helper(self, losses, attr_name):
        threshold_condition = losses > self.threshold
        num_above_threshold = torch.sum(threshold_condition)
        if num_above_threshold >= 1:
            loss = torch.mean(losses[threshold_condition])
        else:
            loss = torch.mean(losses)*0 # set loss to 0
        self.add_to_recordable_attributes(name=attr_name, is_stat=True)
        setattr(self, attr_name, num_above_threshold)
        return loss