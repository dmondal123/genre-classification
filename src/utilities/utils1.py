import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from collections import Counter
import seaborn as sns
import plotly.graph_objects as go

# source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap='Blues', print_values=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = np.unique(classes)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap trace
    heatmap = go.Heatmap(z=cm[::-1],  # Reverse rows to match true label order
                         colorscale=cmap,
                         x=classes,  # Predicted labels
                         y=classes[::-1],  # True labels (reversed order)
                         )

    # Create a layout
    layout = go.Layout(
        title=title if title else "Confusion Matrix",
        xaxis=dict(title="Predicted label"),
        yaxis=dict(title="True label"),
        width=800,  # Adjust the width
        height=800,  # Adjust the height
    )

    # Create a figure and add the heatmap trace
    fig = go.Figure(data=heatmap, layout=layout)

    # Show the plot
    fig.show()

    if print_values:
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                fig.add_annotation(x=classes[j], y=classes[::-1][i],
                                   text=format(cm[i, j], fmt),
                                   showarrow=False,
                                   font=dict(color="white" if cm[i, j] > thresh else "black"))

    if normalize:
        title = title + '_norm' if title else "Confusion Matrix_norm"
    filename = 'plot/' + title + '.pdf' if title else 'plot/Confusion_Matrix.pdf'

class OneHotEncoder:
    def __init__(self, labels):
        self.encoder = LabelBinarizer()
        self.labels = self.encoder.fit_transform(labels)

    def get(self, onehot):
        return self.encoder.inverse_transform(np.array([onehot]))[0]


class labelEncoder:
    def __init__(self, labels):
        self.encoder = LabelEncoder()
        self.labels = self.encoder.fit_transform(labels)

    def get(self, label):
        encoded_label = self.encoder.transform([label])[0]
        return encoded_label

def randcolor():
    return '#{:06x}'.format(randint(0, 256 ** 3))


def plot(vectors, labels, output='img.pdf'):
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(normalize(vectors))

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    scatter = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(
            color=labels,
            colorscale='hls',  # Set the colorscale to 'hls'
            opacity=0.7,  # Set the transparency
            size=8,  # Set the marker size
            colorbar=dict(title='Labels')  # Add a colorbar
        )
    )

    # Create a layout
    layout = go.Layout(
        title="Scatterplot",
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        width=800,  # Adjust the width
        height=800,  # Adjust the height
    )
    
    # Create a figure and add the scatter plot trace
    fig = go.Figure(data=scatter, layout=layout)
    
    # Show the plot
    fig.show()


def clean_and_filter(data, what, min_count=15, sort_by='value', plotting=True):
    """
    Remove the '?' items and keep only the ones
    with more than min_count occurrences
    """
    u = Counter(what)  # map label -> n. occurences

    enough_big = [(m not in ['?', '0'] and u[m] > min_count) for m in what]

    if type(data[0][0]) == np.ndarray:  # list of heterogeneous datasets
        data_filtered = [d[enough_big] for d in data]
    else:
        data_filtered = data[enough_big]
    what_filtered = what[enough_big]

    if sort_by == 'value':
        x = Counter(what_filtered).most_common()
    elif sort_by == 'key':
        x = sorted(Counter(what_filtered).items())
    else:
        x = Counter(what_filtered).items()

    labels, values = zip(*x)
    indexes = np.arange(len(labels))
    width = 1

    if plotting:
        bar_trace = go.Bar(x=[i - 1 + width for i in indexes],y=values,width=width,)

        # Layout adjustments
        layout = go.Layout(
            xaxis=dict(tickvals=[i for i in range(len(indexes))], ticktext=labels),
            yaxis=dict(title='Values'),
            title='No. of songs per Genre',
            width=900,
            height=500,
            margin=dict(l=50, r=50, t=100, b=50),
        )
        
        # Create the figure and plot
        fig = go.Figure(data=[bar_trace], layout=layout)
        fig.show()

    return data_filtered, what_filtered


from imblearn.over_sampling import SMOTE

def extract_balanced(x, y, n_samples=30):
    """
    Create a well-balanced dataset containing n samples for each class using SMOTE.
    """
    # Apply SMOTE to balance the dataset
    smote = SMOTE(sampling_strategy='auto', k_neighbors=min(n_samples, 5))
    x_resampled, y_resampled = smote.fit_resample(x, y)
    
    # Extract the balanced dataset without random selection
    yy = y_resampled
    if isinstance(x_resampled, list):
        xx = [d for d in x_resampled]
    else:
        xx = x_resampled

    return np.array(xx), np.array(yy)