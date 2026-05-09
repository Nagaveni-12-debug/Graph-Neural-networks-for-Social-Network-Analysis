import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        pred = out.argmax(dim=-1)
        
        y_true = data['news'].y[data['news'].test_mask].cpu().numpy()
        y_pred = pred[data['news'].test_mask].cpu().numpy()
        
        print("\n--- Model Evaluation (Test Set) ---")
        print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))
        
        # Confusion Matrix using Matplotlib
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()

if __name__ == "__main__":
    # This would normally be called after training in engine.py
    pass
