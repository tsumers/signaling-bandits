import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter


def heatmap_speaker_action_dataframe(full_utterance_probabilities_df, speaker_one, rewards, speaker_two=None, vmax=None,
                                     vmin=None, cmap="Purples"):
    """Visualize the probability of a speaker choosing an utterance."""

    plt.figure()
    collapsed_df = full_utterance_probabilities_df.groupby(["value", "feature"]).agg(np.mean).reset_index()

    collapsed_df["label"] = collapsed_df.truthful.apply(lambda x: "X" if x else "")

    speaker_one_col = speaker_one.name + "_prob"
    if speaker_two is None:
        to_plot_col = speaker_one_col
        title = "Utterance Probabilities for {} Speaker".format(speaker_one.name)
    else:
        cmap = "PuOr_r"
        title = "Difference in Probabilities: {} (gold) vs {} (purple)".format(speaker_one.name, speaker_two.name)
        collapsed_df["speaker_diff"] = collapsed_df[speaker_one_col] - collapsed_df[speaker_two.name + "_prob"]
        to_plot_col = "speaker_diff"

    utterance_selection = collapsed_df.pivot("value", "feature", to_plot_col)
    labels = collapsed_df.pivot("value", "feature", "label")

    # Sort columns in descending order of features
    features_in_descending_order = rewards[utterance_selection.columns].sort_values(ascending=False).index
    utterance_selection = utterance_selection.reindex(features_in_descending_order, axis=1)
    labels = labels.reindex(features_in_descending_order, axis=1)

    if vmax is None:
        vmax = collapsed_df[to_plot_col].max()
    if vmin is None:
        vmin = collapsed_df[to_plot_col].min()

    formatter = FuncFormatter(_format_positive)

    ax = sns.heatmap(utterance_selection, annot=labels, fmt='', linewidths=.5, cmap=cmap, vmin=vmin, vmax=vmax,
                     cbar_kws={'format': formatter})

    ax.invert_yaxis()

    plt.title(title)


def _format_positive(x, pos):

    return '%0.2f' % abs(x)
