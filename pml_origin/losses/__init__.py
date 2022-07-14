from .base_metric_loss_function import BaseMetricLossFunction, MultipleLosses
from .angular_loss import AngularLoss
from .arcface_loss import ArcFaceLoss
from .circle_loss import CircleLoss
from .contrastive_loss import ContrastiveLoss
from .cosface_loss import CosFaceLoss
from .cross_batch_memory import CrossBatchMemory
from .fast_ap_loss import FastAPLoss
from .generic_pair_loss import GenericPairLoss
from .intra_pair_variance_loss import IntraPairVarianceLoss
from .large_margin_softmax_loss import LargeMarginSoftmaxLoss
from .lifted_structure_loss import LiftedStructureLoss, GeneralizedLiftedStructureLoss
from .margin_loss import MarginLoss
from .multi_similarity_loss import MultiSimilarityLoss
from .nca_loss import NCALoss
from .normalized_softmax_loss import NormalizedSoftmaxLoss
from .ntxent_loss import NTXentLoss
from .n_pairs_loss import NPairsLoss
from .proxy_anchor_loss import ProxyAnchorLoss
from .proxy_losses import ProxyNCALoss
from .signal_to_noise_ratio_losses import SignalToNoiseRatioContrastiveLoss
from .soft_triple_loss import SoftTripleLoss
from .sphereface_loss import SphereFaceLoss
from .triplet_margin_loss import TripletMarginLoss
from .tuplet_margin_loss import TupletMarginLoss
from .weight_regularizer_mixin import WeightRegularizerMixin
