from . import loss_and_miner_utils as lmu, common_functions as c_f
import numpy as np
import torch
import faiss


class MatchFinder:
    def __init__(self, mode="dist", threshold=None):
        assert mode in ["dist", "squared_dist", "sim"]
        self.mode = mode
        self.threshold = threshold

    def operate_on_emb(self, input_func, query_emb, ref_emb=None, *args, **kwargs):
        if ref_emb is None:
            ref_emb = query_emb
        return input_func(query_emb, ref_emb, *args, **kwargs)

    # for a batch of queries
    def get_matching_pairs(self, query_emb, ref_emb=None, threshold=None, return_tuples=False):
        with torch.no_grad():
            threshold = threshold if self.threshold is None else self.threshold
            return self.operate_on_emb(self._get_matching_pairs, query_emb, ref_emb, threshold, return_tuples)

    def _get_matching_pairs(self, query_emb, ref_emb, threshold, return_tuples):
        if self.mode == "dist":
            mat = lmu.dist_mat(query_emb, ref_emb, squared=False)
        elif self.mode == "squared_dist":
            mat = lmu.dist_mat(query_emb, ref_emb, squared=True)
        elif self.mode == "sim":
            mat = lmu.sim_mat(query_emb, ref_emb)
        
        if self.mode == "sim":
            matches = mat >= threshold
        else:
            matches = mat <= threshold
        
        matches = matches.cpu().numpy()

        if return_tuples:
            return list(zip(*np.where(matches)))
        return matches

    # where x and y are already matched pairs
    def is_match(self, x, y, threshold=None):
        threshold = threshold if self.threshold is None else self.threshold
        with torch.no_grad():
            if self.mode == "dist":
                dist = torch.nn.functional.pairwise_distance(x, y)
            elif self.mode == "squared_dist":
                dist = torch.nn.functional.pairwise_distance(x, y) ** 2
            elif self.mode == "sim":
                dist = torch.sum(x*y, dim=1)
            output = dist >= threshold if self.mode == "sim" else dist <= threshold
            if output.nelement() == 1:
                return output.detach().item()
            return output.cpu().numpy()


class FaissIndexer:
    def __init__(self,
                 index=None,
                 emb_dim=None):
        self.index = index
        self.emb_dim = emb_dim

    def train_index(self, vectors):
        self.emb_dim = len(vectors[0])
        self.index = faiss.IndexFlatL2(self.emb_dim)
        self.index.add(vectors)

    def search_nn(self, query_batch, k):
        D, I = self.index.search(query_batch, k)
        return I, D

class InferenceModel:
    def __init__(self,
                 trunk,
                 embedder=None,
                 match_finder=None,
                 normalize_embeddings=True,
                 indexer=None,
                 batch_size=64):
        self.trunk = trunk
        self.embedder = c_f.Identity() if embedder is None else embedder
        self.match_finder = MatchFinder(mode="sim",
                                        threshold=0.9) if match_finder is None else match_finder
        self.indexer = FaissIndexer() if indexer is None else indexer
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

    def train_indexer(self, tensors, emb_dim):
        if isinstance(tensors, list):
            tensors = torch.stack(tensors)

        embeddings = torch.Tensor(len(tensors), emb_dim)
        for i in range(0, len(tensors), self.batch_size):
            embeddings[i:i + self.batch_size] = \
            self.get_embeddings(tensors[i:i + self.batch_size], None)[0]

        self.indexer.train_index(embeddings.cpu().numpy())

    def get_nearest_neighbors(self, query, k):
        if not self.indexer.index or not self.indexer.index.is_trained:
            raise RuntimeError('Index must be trained by running `train_indexer`')

        query_emb, _ = self.get_embeddings(query, None)

        indices, distances = self.indexer.search_nn(query_emb.cpu().numpy(), k)
        return indices, distances

    def get_embeddings(self, query, ref):
        if isinstance(query, list):
            query = torch.stack(query)

        self.trunk.eval()
        self.embedder.eval()
        with torch.no_grad():
            query_emb = self.embedder(self.trunk(query))
            ref_emb = query_emb if ref is None else self.embedder(self.trunk(ref))
        if self.normalize_embeddings:
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            ref_emb = torch.nn.functional.normalize(ref_emb, p=2, dim=1)
        return query_emb, ref_emb

    # for a batch of queries
    def get_matches(self, query, ref=None, threshold=None, return_tuples=False):
        query_emb, ref_emb = self.get_embeddings(query, ref)
        return self.match_finder.get_matching_pairs(query_emb, ref_emb, threshold,
                                                    return_tuples)

    # where x and y are already matched pairs
    def is_match(self, x, y, threshold=None):
        x, y = self.get_embeddings(x, y)
        return self.match_finder.is_match(x, y, threshold)