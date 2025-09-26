from sklearn import datasets
import pandas as pd
import numpy as np
import bqplot.pyplot as plt
import ipywidgets as widgets

from IPython.display import display


def linear_classification():

    iris_X = datasets.load_iris()['data']
    iris_y = datasets.load_iris()['target']
    iris_y[iris_y == 2] = 1

    colors = []
    for i in range(iris_y.shape[0]):
        if (iris_y[i] == 0):
            colors.append('#FF8000')
        else:
            colors.append('#33FFFF')

    scaled = iris_X[:,:2] - np.array([np.mean(iris_X[:,0]), np.mean(iris_X[:,1])])
    scaled = scaled / np.array([np.std(iris_X[:,0]), np.std(iris_X[:,1])])

    ################################################################################    

                                # Scales


    sep2_x_sc = plt.LinearScale(min = -3, max = 3)
    sep2_y_sc = plt.LinearScale(min = -3, max = 3)

    sep2_ax_x = plt.Axis(scale=sep2_x_sc,
                    grid_lines='none',
                    num_ticks = 0,
                    label='Toxic Substance Concentration')

    sep2_ax_y = plt.Axis(scale=sep2_y_sc,
                    orientation='vertical',
                    grid_lines='none',
                    num_ticks = 0,
                    label='Mineral Salt Content')

                              # Scatter plot

    sep2_bar = plt.Scatter(x = scaled[:,0],
                      y = scaled[:,1]-1,
                      colors = colors,
                      default_size = 10,
                      scales={'x': sep2_x_sc, 'y': sep2_y_sc})

                                 # Vector

    w1, w2 = 1.0, 1.0
    w = np.array([w1, w2])

    sep2_vector_line = plt.Lines(x = np.array([0, w1]),
                            y = np.array([0, w2]),
                            colors = ['red', 'red'],
                            scales={'x': sep2_x_sc, 'y': sep2_y_sc})

    sep2_vector_label = plt.Label(x = [w1],
                             y = [w2],
                             text = ['(w1, w2)'],
                             size = [10])

    sep2_vector_plane = plt.Lines(x = [-30*(w2 / np.linalg.norm(w)), 30*(w2 / np.linalg.norm(w))],
                             y = [30*(w1 / np.linalg.norm(w)), -30*(w1 / np.linalg.norm(w))],
                             colors = ['red', 'red'],
                             scales={'x': sep2_x_sc, 'y': sep2_y_sc})

    sep2_f = plt.Figure(marks=[sep2_bar, sep2_vector_line, sep2_vector_label, sep2_vector_plane],
                   axes=[sep2_ax_x, sep2_ax_y],
                   legend_location='bottom-right')

    sep2_f.layout.height = '500px'
    sep2_f.layout.width = '500px'

    display(sep2_f)

    @widgets.interact(
              w1 = widgets.FloatSlider(min=-4, max=4, step=0.01, value=1.0),
              w2 = widgets.FloatSlider(min=-4, max=4, step=0.01, value=1.0))

                      # Fonction qui va interagir avec les widgets

    def h(w1, w2):
        sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
        sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]
        w = np.array([w1, w2])
        sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
        sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]