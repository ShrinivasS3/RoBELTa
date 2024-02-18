import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Custom_roBERTa import RobertaClassifierPooling

df = pd .read_csv('data_mod.csv')

text = df["data"].tolist()
labels = df["win"].tolist()
test = text[5]

MODEL_PARAMS = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epochs": 100,
    "chunk_size": 2048,
    "stride": 510,
    "minimal_chunk_length": 510,
    "pooling_strategy": "mean",
}
model = RobertaClassifierPooling(**MODEL_PARAMS, device="cuda")

model.fit(text, labels, epochs=1)

preds = model.predict_classes(test)

accurate = sum(preds == np.array("1").astype(bool))
accuracy = accurate / 1

print(f"Test accuracy: {accuracy}")