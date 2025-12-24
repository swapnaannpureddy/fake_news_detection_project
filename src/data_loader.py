import pandas as pd
import os
from src.data_utils import clean_text

def load_liar(data_dir, split='train'):
    path = os.path.join(data_dir, "liar", f"liar_{split}.csv")

    try:
        df = pd.read_csv(path)

        if 'claim' in df.columns:
            df['label'] = df['label'].astype(int)
            df['text'] = df['claim'].astype(str).apply(clean_text)
        else:
            raise ValueError("Expected 'claim' column not found.")
    except Exception as e:
        # fallback for legacy CSVs
        df = pd.read_csv(path, header=None)
        df.columns = ["label", "statement"]
        true_labels = {'true', 'mostly-true', 'half-true'}
        df['label'] = df['label'].apply(lambda x: 0 if x in true_labels else 1)
        df['text'] = df['statement'].astype(str).apply(clean_text)

    df['context_feat'] = ''
    return df[['text', 'label', 'context_feat']]

