import numpy as np
import pandas as pd
from sklearn import datasets
import ipywidgets as widgets
from IPython.display import display, HTML

# Détection Colab
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


def linear_classification():
    # === Jeu de données ===
    iris_X = datasets.load_iris()['data']
    iris_y = datasets.load_iris()['target']
    iris_y[iris_y == 2] = 1

    colors = ['#FF8000' if y == 0 else '#33FFFF' for y in iris_y]

    scaled = iris_X[:, :2] - np.array([np.mean(iris_X[:, 0]), np.mean(iris_X[:, 1])])
    scaled = scaled / np.array([np.std(iris_X[:, 0]), np.std(iris_X[:, 1])])

    # === Vérification environnement ===
    if in_colab():
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
        except Exception:
            display(HTML("<b style='color:red'>⚠️ Attention :</b> "
                         "Les widgets interactifs peuvent ne pas fonctionner dans Colab. "
                         "Essaie plutôt dans Jupyter Notebook/Lab pour une meilleure expérience."))

    try:
        import bqplot.pyplot as plt

        # ==== Version BQPlot (Jupyter) ====
        sep2_x_sc = plt.LinearScale(min=-3, max=3)
        sep2_y_sc = plt.LinearScale(min=-3, max=3)

        sep2_ax_x = plt.Axis(scale=sep2_x_sc, label='Toxic Substance Concentration')
        sep2_ax_y = plt.Axis(scale=sep2_y_sc, orientation='vertical', label='Mineral Salt Content')

        sep2_bar = plt.Scatter(x=scaled[:, 0],
                               y=scaled[:, 1]-1,
                               colors=colors,
                               default_size=10,
                               scales={'x': sep2_x_sc, 'y': sep2_y_sc})

        w1, w2 = 1.0, 1.0
        w = np.array([w1, w2])

        sep2_vector_line = plt.Lines(x=[0, w1],
                                     y=[0, w2],
                                     colors=['red'],
                                     scales={'x': sep2_x_sc, 'y': sep2_y_sc})

        sep2_vector_plane = plt.Lines(x=[-30*w2/np.linalg.norm(w), 30*w2/np.linalg.norm(w)],
                                      y=[30*w1/np.linalg.norm(w), -30*w1/np.linalg.norm(w)],
                                      colors=['red'],
                                      scales={'x': sep2_x_sc, 'y': sep2_y_sc})

        fig = plt.Figure(marks=[sep2_bar, sep2_vector_line, sep2_vector_plane],
                         axes=[sep2_ax_x, sep2_ax_y])
        fig.layout.height = '400px'
        fig.layout.width = '400px'

        display(fig)

        @widgets.interact(
            w1=widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0),
            w2=widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0)
        )
        def update(w1, w2):
            w = np.array([w1, w2])
            sep2_vector_line.x = [0, w1]
            sep2_vector_line.y = [0, w2]
            sep2_vector_plane.x = [-30*w2/np.linalg.norm(w), 30*w2/np.linalg.norm(w)]
            sep2_vector_plane.y = [30*w1/np.linalg.norm(w), -30*w1/np.linalg.norm(w)]

    except Exception:
        # ==== Version Matplotlib (Colab fallback) ====
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(scaled[:, 0], scaled[:, 1]-1, c=colors, s=10)
        line, = ax.plot([0, 1], [0, 1], 'r-')

        def update(w1=1.0, w2=1.0):
            w = np.array([w1, w2])
            line.set_xdata([0, w1])
            line.set_ydata([0, w2])
            fig.canvas.draw_idle()

        out = widgets.interactive_output(update, {
            'w1': widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0, description="w1"),
            'w2': widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0, description="w2"),
        })

        sliders = widgets.VBox([
            widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0, description="w1"),
            widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0, description="w2")
        ])

        display(widgets.HBox([sliders, out]))
