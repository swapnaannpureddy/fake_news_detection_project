import matplotlib.pyplot as plt
import numpy as np

def plot_attention(text, attn_weights, save_path=None):
    tokens = text.split()
    attn = attn_weights[:len(tokens)].detach().cpu().numpy()

    # ✅ Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 1))
    cax = ax.imshow([attn], cmap='hot', aspect='auto')
    ax.set_yticks([])
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    fig.colorbar(cax)

    # ✅ Save only if a path is provided
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig  # ✅ IMPORTANT: this line MUST exist


def highlight_text(text, attn_weights, threshold=0.15):
    tokens = text.split()
    attn = attn_weights[:len(tokens)].detach().cpu().numpy()
    highlighted = [
        f"**{t}**" if a > threshold else t
        for t, a in zip(tokens, attn)
    ]
    return ' '.join(highlighted)

def lime_explain(text, model, tokenizer, class_names=['REAL', 'FAKE']):
    from lime.lime_text import LimeTextExplainer
    import torch

    def predict_fn(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_fn, num_features=10, labels=[0, 1])
    return exp

def shap_explain(texts, model, tokenizer):
    import shap
    import torch

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    e = shap.Explainer(
        lambda x: model(x['input_ids'], x['attention_mask'])[0].softmax(dim=1).detach().cpu().numpy(),
        tokenizer
    )
    shap_values = e(texts)
    return shap_values
