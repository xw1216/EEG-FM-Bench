from enum import Enum

class TrainStage(Enum):
    PRETRAIN = 'pretrain'
    FINETUNE = 'finetune'
    EVAL = 'eval'
    ALL = 'all'


class PriorType(Enum):
    HYBRID = 'hybrid'
    REGION = 'region'
    NETWORK = 'network'
    NONE = 'none'


class TemporalConvType(Enum):
    STRIDE = 'stride'
    MULTISCALE = 'multiscale'


class StdFactorType(Enum):
    CURRENT_DEPTH = 'current_depth'
    GLOBAL_DEPTH = 'global_depth'
    DIM_RATIO = 'dim_ratio'
    DISABLED = 'disabled'


class PretrainTaskListType(Enum):
    ALL = 'all'
    GPT = 'gpt'
    MAE = 'mae'

class PretrainSuperviseType(Enum):
    DEFAULT = 'default'
    CLASSIFICATION = 'classification'


class EncoderTaskType(Enum):
    CLS     = 0
    GPT     = 1
    MAE_TP  = 2
    MAE_CH  = 3


class DatasetTaskType(Enum):
    UNKNOWN                 = 4
    CLINICAL                = 5
    MOTOR_IMAGINARY         = 6
    MOTOR_EXECUTION         = 7
    EMOTION                 = 8
    SEIZURE                 = 9
    SLEEP_STAGE             = 10
    RESTING                 = 11
    WORKLOAD                = 12
    ARTIFACT                = 13
    LINGUAL                 = 14
    VISUAL                  = 15
    AUDIO                   = 16
    ERP                     = 17
    # MULTI                   = 18
