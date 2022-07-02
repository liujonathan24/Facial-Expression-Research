import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def heatmap(data,
            size_scale: int = 75,
            figsize: tuple = (10, 8),
            x_rotation: int = 270,
            cmap=mpl.cm.RdBu
            ):
    """
    Build a heatmap based on the given data frame (Pearson's correlation).
    Args:
        data: the dataframe to plot the cross-correlation of
        size_scale: the scaling parameter for box sizes
        figsize: the size of the figure
    Returns:
        None
    Notes:
        based on -
        https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    """


    # copy the data before mutating it
    data = data.copy()
    # change datetimes and timedelta to floating points
    for column in data.select_dtypes(include=[np.datetime64, np.timedelta64]):
        data[column] = data[column].apply(lambda x: x.value)

    # calc the correlation matrix
    data = data.corr()
    data = pd.melt(data.reset_index(), id_vars='index')
    data.columns = ['x', 'y', 'value']
    x = data['x']
    y = data['y']
    # the size is the absolut value (correlation is on [-1, 1])
    size = data['value'].abs()
    norm = (data['value'] + 1) / 2

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix for the FER-2013 Dataset')
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
    # ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

    im = ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        marker='s',  # Use square as scatterplot marker
        c=norm.apply(cmap)
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=270, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    # move the points from the center of a grid point to the center of a box
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    # move the ticks to correspond with the values
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xticklabels(["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"])
    ax.set_yticklabels(["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"])

    '''labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Anger'
    labels[1] = "Contempt"
    labels[2] = "Disgust"
    labels[3] = "Fear"
    labels[4] = "Happiness"
    labels[5] = "Neutral"
    labels[6] = "Sadness"
    labels[7] = "Surprise"'''
    # = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

    # move the starting point of the x and y axis forward to remove
    # extra spacing from shifting grid points
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # add a color bar legend to the plot
    bar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.outline.set_edgecolor('grey')
    bar.outline.set_linewidth(1)
    bar.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])

data = pd.DataFrame([[1.0, 0.0, 0.14432989690721648, 0.3754295532646048, 0.15034364261168384, 0.2190721649484536, 0.32646048109965636, 0.09621993127147767], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.719626168224299, 0.0, 0.04205607476635514, 1.0, 0.5046728971962616, 0.411214953271028, 0.6869158878504673, 0.48130841121495327], [0.09138095888539817, 0.0, 0.009424941947821336, 0.09206392569321131, 1.0, 0.07813140281382325, 0.09752766015571643, 0.05204207075536129], [0.34543010752688175, 0.0, 0.044623655913978495, 0.4706989247311828, 0.2129032258064516, 1.0, 0.5833333333333334, 0.1913978494623656], [0.4262055528494886, 0.0, 0.033122260107160253, 0.4987822698490015, 0.12907939600584512, 0.5294690696541646, 1.0, 0.07793472966390648], [405.47628698824485, 0.0, 0.017430077016619375, 0.3198216457235509, 0.08025942440210783, 0.15362788812322659, 0.10660721524118362, 1.0]])
map = heatmap(data)
plt.show()