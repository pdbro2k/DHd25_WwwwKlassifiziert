import pandas as pd

class StepSegmenter:

    # constructor
    def __init__(self):
        self.__column = "step"

    # getter
    def get_column(self) -> str:
        return self.__column

    # main method
    def add_steps(self, df: pd.DataFrame) -> pd.DataFrame:
        current_df = df.copy()
        segment_len_df = df[["source", "text"]].groupby("source").count().copy()
        
        current_df[self.__column] = 0
        current_id = -1
        current_step = 0
        step_size = 0
        for i, row in current_df.iterrows():
            if row["source"] != current_id:
                current_id = row["source"]
                step_size = segment_len_df.loc[current_id].values[0]
                current_step = 0
            else:
                current_step += 1
            current_df.at[i, self.__column] = current_step * 100 // step_size

        n = 10
        current_df[self.__column] = n * round(current_df[self.__column] // n) + n // 2
        return current_df
