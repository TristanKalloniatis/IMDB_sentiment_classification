from torch.cuda import is_available

VOCAB_SIZE = 10000
TOKENIZER = "spacy"
BATCH_SIZE = 8
TRAIN_VALID_SPLIT = 0.1
PATIENCE = 5
STATISTICS_FILE = 'statistics.csv'
USE_CUDA = is_available()