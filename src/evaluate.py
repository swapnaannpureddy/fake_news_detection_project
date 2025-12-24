import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def evaluate(model, dataloader, config):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_args = [batch['input_ids'].to(config.DEVICE), batch['attention_mask'].to(config.DEVICE)]
            
            if 'x_bilstm' in batch:  # For ensemble models
                input_args.append(batch['x_bilstm'].to(config.DEVICE))

            labels = batch['labels'].to(config.DEVICE)

            # Get model output
            output = model(*input_args)
            logits = output[0] if isinstance(output, tuple) else output

            # Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # Only positive class probs

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0  # fallback if only one class present

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "roc_auc": roc_auc,
    }
