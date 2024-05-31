import numpy as np
from PSTAAP import load_model, data_preprocess
import shap
import torch
import matplotlib.pyplot as plt

# Load your model, assuming the model file is 'model.pth'
model = load_model('ckpt/Adam_lr0.001_weightdecay1e-06_epochs300.pth')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example feature data
X_train, targets_train, X_test, targets_test, train_loader, X_ori, Y_ori, Y_label = data_preprocess(sample_strategy=1)

def predict_proba(X):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X)
        probs = torch.sigmoid(outputs)
    return probs.cpu().numpy()
# Convert training data to numpy array
X_test = X_test.cpu().numpy()

"""
# Creating a SHAP explainer
K = 100  # Choose 100 samples as representatives
X_sample = shap.sample(X_test, K)
print(X_sample.shape)
"""

explainer = shap.KernelExplainer(predict_proba, X_test)
print(explainer.expected_value)

# Predict probability for the sample
print(predict_proba(X_test)[99])

shap_values = explainer.shap_values(X_test[99])
shap_values = np.array(shap_values)
print(shap_values.shape)

shap.initjs()

# Ensure the sample is correctly selected and formatted
sample = X_test[99]

for i in range(shap_values.shape[0]):
    expl = shap.Explanation(values=shap_values[i], base_values=explainer.expected_value[i], data=sample)
    shap.plots.waterfall(expl)

