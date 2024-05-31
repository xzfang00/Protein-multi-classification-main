import torch
import torch.nn as nn
import torch.nn.init as init
from DataProcess import load_data,  make_ylabel
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, hamming_loss
import numpy as np
import os
from torchviz import make_dot
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk



class MultiLabelCNN(nn.Module):
    def __init__(self, input_dim=46, num_classes=4):
        super(MultiLabelCNN, self).__init__()


        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.fc_layer = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:

            init.xavier_uniform_(m.weight)
            if m.bias is not None:

                init.constant_(m.bias, 0.0)

    def forward(self, x):
        '''
        x: 输入形状为 (batch_size, 1, input_dim)
        '''
        x = x.unsqueeze(1)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        return x


def load_model(model_path):
    model = MultiLabelCNN()
    if os.path.exists(model_path):
        # model=torch.load(model_path)
        # for name, param in model.items():
        #     print(f"Parameter name: {name}, Size: {param.size()}")
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        print("[ERROR]\tModel not found at the specified path.")
        return None


def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)


def browse_model(entry):
    file_path = filedialog.askopenfilename(filetypes=[("PTH files", "*.pth")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)


def predict_threshold(model, inputs, threshold=0.5):
    model.eval()
    inputs = inputs
    with torch.no_grad():
        test_outputs = model(inputs)
        probs = torch.sigmoid(test_outputs)
        binary_preds = (probs >= threshold).float()
    return binary_preds


def map_labels(vector):
    labels = ["A", "C", "M", "S"]
    mapped_labels = [labels[i] for i, val in enumerate(vector) if val == 1]
    return ",".join(mapped_labels)



def display_results(results, targets=None):

    result_window = tk.Toplevel(root)
    result_window.title("Prediction Results")


    tree = ttk.Treeview(result_window)
    if targets!=None:
        tree["columns"] = ("results", "targets")


        tree.heading("#0", text="Sample")
        tree.heading("results", text="Results")
        tree.heading("targets", text="Targets")


        scroll = ttk.Scrollbar(result_window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)


        for idx, (sample_result, target) in enumerate(zip(results, targets)):
            mapped_result = map_labels(sample_result)
            mapped_target = map_labels(target)
            tree.insert("", idx, text=f"Sample {idx + 1}", values=(mapped_result, mapped_target))

    else:
        tree["columns"] = ("results")


        tree.heading("#0", text="Sample")
        tree.heading("results", text="Results")


        scroll = ttk.Scrollbar(result_window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)


        for idx, (sample_result, target) in enumerate(results):
            mapped_result = map_labels(sample_result)
            tree.insert("", idx, text=f"Sample {idx + 1}", values=[mapped_result])


    label_info = tk.Label(result_window, text="A:acetyllysine ,C:crotonyllysine ,M:methyllysine ,S:succinyllysine ")
    label_info.grid(row=1, column=0, columnspan=2)


    tree.grid(row=0, column=0, sticky="nsew")
    scroll.grid(row=0, column=1, sticky="ns")
    result_window.grid_rowconfigure(0, weight=1)
    result_window.grid_columnconfigure(0, weight=1)


    for col in tree["columns"]:
        tree.heading(col, anchor=tk.CENTER)
        tree.column(col, anchor=tk.CENTER)


    result_window.update()


def integrate_and_predict():
    test_file_path = test_entry.get()
    model_file_path = model_entry.get()


    X_test, targets_test = load_data(test_path=test_file_path)

    model = load_model(model_file_path)

    res=predict_threshold(model=model,inputs=X_test)
    print(targets_test)
    if len(targets_test)!=0:
        targets_test=make_ylabel(targets_test)
    else:
        targets_test=None
    display_results(res,targets_test)



# 创建GUI界面
root = tk.Tk()
root.title("MultiLabelCNN Prediction")

# 添加测试集文件路径选择框
test_label = tk.Label(root, text="Choose .mat Test Set:")
test_label.grid(row=0, column=0)
test_entry = tk.Entry(root, width=100)
test_entry.grid(row=0, column=1)
test_button = tk.Button(root, text="Browse", command=lambda: browse_file(test_entry))
test_button.grid(row=0, column=2)

# 添加模型文件路径选择框
model_label = tk.Label(root, text="Choose .pth Model:")
model_label.grid(row=1, column=0)
model_entry = tk.Entry(root, width=100)
model_entry.grid(row=1, column=1)
model_button = tk.Button(root, text="Browse", command=lambda: browse_model(model_entry))
model_button.grid(row=1, column=2)

# 添加预测按钮
predict_button = tk.Button(root, text="Predict", command=integrate_and_predict)
predict_button.grid(row=2, column=1)

root.mainloop()
