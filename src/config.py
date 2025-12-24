import torch
import os
class Config:

    # Base directory: path to project/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Path to data
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_CSV = os.path.join(DATA_DIR, 'liar_train.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'liar_test.csv')

    # Path to saved_models/
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
    MODEL_DIR = SAVED_MODELS_DIR  # ✅ Add this line to avoid AttributeError
    MODEL_SAVE_PATH = os.path.join(SAVED_MODELS_DIR, 'best_model.pt')
    BILSTM_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'bilstm_model.pt')
    TOKENIZER_PATH = os.path.join(SAVED_MODELS_DIR, 'tokenizer')  # Optional if saving tokenizer

    # Model / training parameters
    MAX_LEN = 64
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    VOCAB_SIZE = 20000  # For BiLSTM models
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_CLASSES = 2
    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
     # ✅ Add this for BERT model
    BERT_NAME = 'distilbert-base-uncased'



