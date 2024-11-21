import torch


def accuracy(labels, outputs):
    # print("Rango de etiquetas (min, max):", labels.min(), labels.max())
    preds = outputs.argmax(-1)
    # acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    acc = preds.eq(labels.view_as(preds)).sum().item()
    # print("Predicciones:", preds[:10])
    # print("Etiquetas reales:", labels[:10])
    return acc
    

def save_model(model, path):
    torch.save(model.state_dict(), path)