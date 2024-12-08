from openTSNE import TSNE
import openTSNE.callbacks
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib as mpl

class ProgressCallback(openTSNE.callbacks.Callback):
    def __init__(self, pbar: tqdm, step: int=1) -> None:
        super().__init__()
        self.pbar = pbar
        self.step = step

    def __call__(self, iteration, error, embedding):
        self.pbar.update(self.step)
        return False
    

def visualize_tsne(features: np.ndarray, labels: np.ndarray, label_names: list[str]=None,
                   figsize=(10, 10), dimension=2, perplexity=30, legend_nrow=2):
    
    print(f'{features.shape=}, {labels.shape=}')

    with tqdm(total=750) as pbar:
        tsne = TSNE(n_jobs=8, 
                    n_components=dimension, 
                    perplexity=perplexity, 
                    callbacks_every_iters=1,
                    callbacks=ProgressCallback(pbar, 1))
        trained = tsne.fit(features)

    cluster = np.array(trained)

    print('t-SNE computed, waiting for plot...')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot() if dimension < 3 else fig.add_subplot(projection='3d')
    
    classes = np.unique(labels)
    ncls = len(classes)//2
    for i in classes:
        idx = np.where(labels == i)
        ax_args = dict(
            marker = 'o' if i < ncls else '^', 
            label = i if label_names is None else label_names[int(i)], 
            edgecolors = 'face' if i<10 else '#000000bb', 
            linewidths = 0.5,
            c=mpl.color_sequences['tab10'][int(i%ncls)]
        )

        if dimension < 3:
            ax.scatter(cluster[idx, 0], cluster[idx, 1], **ax_args)
        else:
            ax.scatter(cluster[idx, 0], cluster[idx, 1] ,cluster[idx, 2], **ax_args)
            
    ax.autoscale()

    plt.legend(loc='lower center', ncol=len(classes)//legend_nrow, bbox_to_anchor=(0.5, -0.05))
    plt.axis('off')
    plt.show()

    return cluster, fig

