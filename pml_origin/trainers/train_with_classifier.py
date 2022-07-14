#! /usr/bin/env python3

from .metric_loss_only import MetricLossOnly

import torch
class TrainWithClassifier(MetricLossOnly):

    def compute_embeddings(self, data, prototypes, epoch, labels):
        #print('test2')
        trunk_output,loss,rep = self.get_trunk_output(data,self.prototypes, self.epoch, labels)
        embeddings = self.get_final_embeddings(trunk_output)
        return embeddings,loss,rep

    def calculate_loss(self, curr_batch):
        #print('test')
        data, labels = curr_batch
        #print(self.prototypes)
        #print(self.epoch)
        embeddings,loss,rep = self.compute_embeddings(data, self.prototypes, self.epoch, labels)
        logits = self.maybe_get_logits(embeddings)
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        #print('test')
        loss = torch.mean(loss)
        #print(loss[0])
        #print(loss[1])
        self.losses["metric_loss"] = self.maybe_get_metric_loss(embeddings, labels, indices_tuple)+0.1*loss
        self.losses["classifier_loss"] = 0.4*self.maybe_get_classifier_loss(logits, labels)

    def maybe_get_classifier_loss(self, logits, labels):
        if logits is not None:
            return self.loss_funcs["classifier_loss"](logits, labels.to(logits.device))
        return 0

    def maybe_get_logits(self, embeddings):
        if self.models.get("classifier", None) and self.loss_weights.get("classifier_loss", 0) > 0:
            return self.models["classifier"](embeddings)
        return None

    def allowed_model_keys(self):
        return super().allowed_model_keys()+["classifier"]

    def allowed_loss_funcs_keys(self):
        return super().allowed_loss_funcs_keys()+["classifier_loss"]
