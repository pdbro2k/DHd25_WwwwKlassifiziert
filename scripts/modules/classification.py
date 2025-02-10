import pandas as pd
from torch import device
from transformers import pipeline
from typing import Union

class TernarySentimentClassifier:

    # constructor
    def __init__(
        self, 
        model_path: str, 
        device: Union[int, str, device]
    ):
        self.__model_path = model_path
        self.__device = device
        self.__classify = pipeline("text-classification", model=model_path, device=device)

    # getters
    def get_model_path(self) -> str:
        return self.__model_path
    
    def get_device(self) -> Union[int, str, device]:
        return self.__device

    # main methods
    def classify(self, text: str) -> dict:
        return self.__classify(text)[0]
    
    def add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        current_df = df.copy()
        current_df["label"] = current_df["text"].apply(lambda x: self.__classify(x)["label"])
        return current_df
    
    def add_polarities(self, df: pd.DataFrame) -> pd.DataFrame:
        if not "label" in df.columns:
            current_df = self.add_labels(df)
        else:
            current_df = df.copy()
        current_df["polarity"] = current_df["label"].apply(lambda x: get_polarity(x))
        return current_df
    
# "static functions"
def get_polarity(label: str) -> int:
    if label == "positive":
        return 1
    if label == "negative":
        return -1
    return 0

def get_hsl(label: str, score: float) -> str:
    if label == "positive":
        hue = 120
    else:
        hue = 0
    if label == "neutral":
        sat = 0
    else:
        sat = 75
    lig = 100 - int(50 * score)
    return f"hsl({hue}, {sat}%, {lig}%)"