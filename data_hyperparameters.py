from torch.cuda import is_available

VOCAB_SIZE = 10000
TOKENIZER = "spacy"
BATCH_SIZE = 32
TRAIN_VALID_SPLIT = 0.1
PATIENCE = 5
STATISTICS_FILE = 'statistics.csv'
USE_CUDA = is_available()
STORE_DATA_ON_GPU_IF_AVAILABLE = False
EMBEDDING_DIMENSION = 100
HIDDEN_SIZE = 32
DROPOUT = 0.25
INTER_RECURRENT_LAYER_DROPOUT = 0.25
INTRA_RECURRENT_LAYER_DROPOUT = 0.
EMBEDDING_DROPOUT = 0.25
NUM_CONV_FEATURES = 100
EPOCHS = 10
MIN_REVIEW_LENGTH = 10
CONTEXT_SIZE = 3
UNIGRAM_DISTRIBUTION_POWER = 0.75
NUM_NEGATIVE_SAMPLES = 10
INNER_PRODUCT_CLAMP = 4.
SUBSAMPLE_THRESHOLD = 1e-4
WORD_EMBEDDING_DIMENSION = 20
VERBOSE_LOGGING = False
TRAIN_PROPORTION = 0.9
WORD_EMBEDDING_BATCH_SIZE = 128
WORD_EMBEDDING_EPOCHS = 5
REPORT_ACCURACY_EVERY = 5
TRANSFORMER_MAX_LEN = 500
TRANSFORMER_NHEAD = 4
TRANSFORMER_DIM_FEEDFORWARD = 512
POSITIONAL_ENCODING_DROPOUT = 0.2
TRANSFORMER_DROPOUT = 0.4
