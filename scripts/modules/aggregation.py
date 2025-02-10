import math

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from typing import List, Optional, Tuple, Union
from pandas._typing import AggFuncType

class PolarityAggregator:

    # constructor
    def __init__(
        self, 
        strategy: Union[str, AggFuncType] = "mean", 
        xlim: Tuple[float, float] = (0, 100), 
        neutral_lim: Tuple[float, float] = (-.05, .05)
    ):
        self.__x = "step"
        self.__y = "polarity"

        self.__strategy = strategy # TODO: validate

        self.__xlim = xlim # TODO: validate
        self.__ylim = (-1, 1.1)

        self.__neutral_lim = neutral_lim

    # getters
    def get_x(self) -> str:
        return self.__x
    
    def get_y(self) -> str:
        return self.__y
    
    def get_strategy(self) -> Union[str, AggFuncType]:
        return self.__strategy
    
    def get_xlim(self) -> Tuple[float, float]:
        return self.__xlim
    
    def get_ylim(self) -> Tuple[float, float]:
        return self.__ylim
    
    def get_neutral_lim(self) -> Tuple[float, float]:
        return self.__neutral_lim

    # main methods
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(["source", self.__x]).agg({self.__y: self.__strategy}).reset_index().copy()
    
    def get_single(
        self, 
        df: pd.DataFrame, 
        idx: str
    ) -> pd.DataFrame:
        return df.loc[df["source"] == idx].copy()
    
    def get_steps(
        self, 
        df: pd.DataFrame, 
        steps: List[int]
    ) -> pd.DataFrame:
        return df.loc[df[self.__x].isin(steps)].copy()
    
    def plot(
        self, 
        df: pd.DataFrame, 
        title: str = "", 
        ax: Optional[Axes] = None
    ) -> Axes:
        ax = sns.lineplot(self.aggregate(df), x=self.__x, y=self.__y, ax=ax)
    
        ax.set(xlim=self.__xlim, ylim=self.__ylim, xlabel=self.__x, ylabel=self.__y)
        ax.title.set_text(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # add vertical bar to mark neutral area
        min_neutral = self.__neutral_lim[0]
        max_neutral = self.__neutral_lim[1]
        ax.axhspan(ymin=min_neutral, ymax=max_neutral, zorder=-100, color="lightgrey", alpha=.5)
        return ax
    
    def plot_multiple(
        self, df: pd.DataFrame, 
        ids: List[str], 
        nrows: int, 
        figsize: Tuple[int, int]
    ) -> Tuple[Figure, Axes]:
        ncols = math.ceil(len(ids) / nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)

        for i, idx in enumerate(ids):
            current_df = self.get_single(df, idx)
            self.plot(current_df, idx, axes[i // ncols, i % ncols])
            
        if len(ids) % ncols > 0:
            for i in range(len(ids) % ncols, ncols):
                axes[nrows - 1, i].axis("off")
            
        fig.subplots_adjust(top=1.25, right=1.1)
        return fig, axes