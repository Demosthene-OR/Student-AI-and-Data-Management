from sklearn import datasets
import pandas as pd
import ipyvolume as ipv
import numpy as np
import ipywidgets as widgets
import bqplot as bq
import bqplot.pyplot as plt
from IPython.display import display, Markdown
import time
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import markdown


def regression_widget():
    ################################################################################   

                            # Simulation of Data

    linreg1_n_samples = 400
    alpha = 0.5
    bias = 1.5
    x_min = -4
    x_max = 4
    noise_var = 1

    data = np.vstack([np.linspace(x_min, x_max, linreg1_n_samples),
                      alpha * np.linspace(-4, 4, linreg1_n_samples) + bias + np.random.normal(0, noise_var, linreg1_n_samples)]).T


    colors = ["blue"] * linreg1_n_samples          

                                # Scales

    from IPython.display import display

    linreg1_x_sc = plt.LinearScale(min = x_min, max = x_max)
    linreg1_y_sc = plt.LinearScale(min = x_min, max = x_max)

    linreg1_ax_x = plt.Axis(scale=linreg1_x_sc,
                    grid_lines='none',
                    label='x')

    linreg1_ax_y = plt.Axis(scale=linreg1_y_sc,
                    orientation='vertical',
                    grid_lines='none',
                    label='y')

                              # Scatter plot of data

    linreg1_bar = plt.Scatter(x = data[:,0],
                           y = data[:,1],
                           colors = colors,
                           default_size = 10,
                           scales={'x': linreg1_x_sc, 'y': linreg1_y_sc})

                              # Linear Regression

    w1 = 1
    linreg1_vector_plane = plt.Lines(x = [-30, 30],
                                  y = [-30*w1, 30*w1],
                                  colors = ['red', 'red'],
                                  scales= {'x': linreg1_x_sc, 'y': linreg1_y_sc})

    linreg1_f = plt.Figure(marks=[linreg1_bar, linreg1_vector_plane],
                        axes=[linreg1_ax_x, linreg1_ax_y],
                        title='Linear Regression',
                        legend_location='bottom-right',
                        animation_duration = 0)

    linreg1_f.layout.height = '400px'
    linreg1_f.layout.width = '400px'

                            # Widgets

    beta_hat = widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0, description = markdown.markdown(r"$\beta_1$"))
    bias_hat = widgets.FloatSlider(min=-4, max=4, step=0.1, value=0, description = markdown.markdown(r"$\beta_0$"))
    true_bias = widgets.FloatSlider(min=-4, max=4, step=0.5, value=1.5, description = markdown.markdown(r"$\alpha_0$"))
    true_coef = widgets.FloatSlider(min=-4, max=4, step=0.5, value=0.5, description = markdown.markdown(r"$\alpha_1$"))
    true_variance = widgets.FloatSlider(min=0, max=1.5, step=0.1, value=1.0, description = markdown.markdown(r"$\sigma$"))

                      # Fonction qui va interagir avec les widgets

    beta_hat.style = {'description_width': '20px', 'width' : '80%'}
    bias_hat.style = {'description_width': '20px', 'width' : '80%'}
    true_coef.style = {'description_width': '20px'}
    true_bias.style = {'description_width': '20px'}
    true_variance.style = {'description_width': '20px'}


    def linreg1_update_regression(args):
        linreg1_vector_plane.y = [-30*beta_hat.value + bias_hat.value, 30*beta_hat.value + bias_hat.value] 

    #def linreg1_update_data(args):
    #    alpha = true_coef.value
    #    noise_var = true_variance.value
    #    
    #    data = np.vstack([np.linspace(x_min, x_max, linreg1_n_samples),
    #                      alpha * np.linspace(-4, 4, linreg1_n_samples) + true_bias.value + np.random.normal(0, noise_var, linreg1_n_samples)]).T
    #    linreg1_bar.x = data[:, 0]
    #    linreg1_bar.y = data[:, 1]

    beta_hat.observe(linreg1_update_regression)
    bias_hat.observe(linreg1_update_regression)
    #true_coef.observe(linreg1_update_data)
    #true_bias.observe(linreg1_update_data)
    #true_variance.observe(linreg1_update_data)

    #data_sliders = widgets.VBox([true_coef, true_bias, true_variance])
    regression_sliders = widgets.VBox([beta_hat, bias_hat])
    #tab = widgets.Tab([regression_sliders, data_sliders])
    tab = widgets.Tab([regression_sliders])
    tab.set_title(0, 'Regression')
    #tab.set_title(1, 'Data')
    widget = widgets.HBox([linreg1_f, tab])
    widget.layout.align_items = 'center'
    display(widget)
    return



def interactive_MSE():

    ################################################################################   

                            # Simulation of Data
    np.random.seed(3)
    linreg3_n_samples = 20
    alpha = 0.7
    x_min = -4
    x_max = 4
    noise_var = 1.2

    linreg3_data = np.vstack([np.linspace(x_min, x_max, linreg3_n_samples),
                      alpha * np.linspace(-4, 4, linreg3_n_samples) + np.random.normal(0, noise_var, linreg3_n_samples)]).T


    colors = ["blue"] * linreg3_n_samples       

    xx = linreg3_data[:, 0]
    yy = linreg3_data[:, 1]
    distances_x = []
    distances_y = []
    for x,y in zip(xx, yy):
        distances_x.append(x)
        distances_y.append(1.5*x)

        distances_x.append(x)
        distances_y.append(y)

        distances_x.append(x)
        distances_y.append(1.5*x)

    def linreg3_computeloss(beta):
        return np.linalg.norm(linreg3_data[:, 1] - linreg3_data[:, 0] * beta)**2 / linreg3_n_samples


                                # Scales

    from IPython.display import display

    linreg3_x_sc = plt.LinearScale(min = x_min, max = x_max)
    linreg3_y_sc = plt.LinearScale(min = x_min, max = x_max)

    linreg3_loss_x_sc = plt.LinearScale(min = -4, max = 4)
    linreg3_loss_y_sc = plt.LinearScale(min = 0, max = 100)

    linreg3_loss_ax_x = plt.Axis(scale=linreg3_loss_x_sc,
                                 grid_lines='none',
                                 label='x')

    linreg3_loss_ax_y = plt.Axis(scale=linreg3_loss_y_sc,
                                 orientation='vertical',
                                 grid_lines='none',
                                 label='x')

    linreg3_ax_x = plt.Axis(scale=linreg3_x_sc,
                            grid_lines='none',
                            label='x')

    linreg3_ax_y = plt.Axis(scale=linreg3_y_sc,
                            orientation='vertical',
                            grid_lines='none',
                            label='y')

                              # Scatter plot of data

    linreg3_bar = plt.Scatter(x = linreg3_data[:,0],
                              y = linreg3_data[:,1],
                              colors = colors,
                              default_size = 10,
                              scales={'x': linreg3_x_sc, 'y': linreg3_y_sc})

    linreg3_distances = plt.Lines(x = distances_x,
                                  y = distances_y,
                                  colors = ['green'] ,
                                  scales = {'x': linreg3_x_sc, 'y': linreg3_y_sc})

                              # Linear Regression

    w1 = 1
    linreg3_vector_plane = plt.Lines(x = [-30, 30],
                                     y = [-30 * 1.5, 30 * 1.5],
                                     colors = ['red', 'red'],
                                     scales= {'x': linreg3_x_sc, 'y': linreg3_y_sc})

    linreg3_pan_zoom = bq.interacts.PanZoom(allow_pan = True,
                                            allow_zoom = True,
                                            scales = {'x': [linreg3_x_sc], 'y': [linreg3_y_sc]} )

    linreg3_f = plt.Figure(marks=[linreg3_distances, linreg3_bar, linreg3_vector_plane],
                           axes=[linreg3_ax_x, linreg3_ax_y],
                           #interaction = linreg3_pan_zoom,
                           title='Linear Regression',
                           legend_location='bottom-right',
                           animation_duration = 0)


    linreg3_f.layout.height = '400px'
    linreg3_f.layout.width = '400px'

                            # Widgets

    linreg3_beta_hat = widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.5, description = markdown.markdown(r"$\beta_1$"))
    linreg3_loss = widgets.HTML(value="<p style='font-size:20px'>MSE = {}</p>".format(np.round(linreg3_computeloss(1.5), 2)),
                                layout = widgets.Layout(min_width = '300px'))

    linreg3_beta_hat.style = {'description_width': '20px', 'width' : '80%'}
    linreg3_loss.style = {"font_size" : '30px'}


    def linreg3_update_regression(args):
        linreg3_vector_plane.y = [-30*linreg3_beta_hat.value, 30*linreg3_beta_hat.value]

    def linreg3_update_distances(args):
        xx = linreg3_data[:, 0]
        yy = linreg3_data[:, 1]
        beta = linreg3_beta_hat.value
        distances_x = []
        distances_y = []
        for x,y in zip(xx, yy):
            distances_x.append(x)
            distances_y.append(beta*x)

            distances_x.append(x)
            distances_y.append(y)

            distances_x.append(x)
            distances_y.append(beta*x)

        linreg3_distances.x = distances_x
        linreg3_distances.y = distances_y

    def linreg3_update_loss(args):
        loss = np.round(linreg3_computeloss(linreg3_beta_hat.value), 2)
        linreg3_loss.value = "<p style='font-size:20px; color:green'>MSE = {}</p>".format(loss)






    linreg3_beta_hat.observe(linreg3_update_regression)
    linreg3_beta_hat.observe(linreg3_update_distances)
    linreg3_beta_hat.observe(linreg3_update_loss)

    plot = widgets.HBox([linreg3_f, linreg3_loss], layout = widgets.Layout())
    widget = widgets.VBox([plot, linreg3_beta_hat])
    plot.layout.align_items = 'center'
    widget.layout.align_items = 'center'
    display(widget)



def polynomial_regression():
    ################################################################################

    # Simulation of Data

    polyreg1_n_samples = 400
    polyreg1_alpha_1 = 1
    polyreg1_alpha_2 = - 0.1
    polyreg1_bias = 1.5
    polyreg1_x_min = 0
    polyreg1_x_max = 10
    polyreg1_noise_var = 1

    polyreg1_x = np.linspace(polyreg1_x_min, polyreg1_x_max,polyreg1_n_samples)
    polyreg1_data = np.vstack([polyreg1_x,
                               polyreg1_alpha_2 * (polyreg1_x**2)\
                               + polyreg1_alpha_1 * polyreg1_x   \
                               + polyreg1_bias                   \
                               + np.random.normal(0, polyreg1_noise_var, polyreg1_n_samples)]).T


    # Regression parameters

    polyreg1_beta_0 = 0
    polyreg1_beta_1 = 0.5
    polyreg1_beta_2 = 1

    polyreg1_reg = polyreg1_beta_0 + polyreg1_beta_1 * polyreg1_x + polyreg1_beta_2 * (polyreg1_x**2)


    polyreg1_colors = ["blue"] * polyreg1_n_samples

    # Scales


    polyreg1_x_sc = plt.LinearScale(min=polyreg1_x_min, max=polyreg1_x_max)
    polyreg1_y_sc = plt.LinearScale(min=polyreg1_x_min, max=polyreg1_x_max)

    polyreg1_ax_x = plt.Axis(scale=polyreg1_x_sc,
                             grid_lines='none',
                             label='x')

    polyreg1_ax_y = plt.Axis(scale=polyreg1_y_sc,
                             orientation='vertical',
                             grid_lines='none',
                             label='y')

    # Scatter plot of data

    polyreg1_bar = plt.Scatter(x=polyreg1_data[:, 0],
                               y=polyreg1_data[:, 1],
                               colors=polyreg1_colors,
                               default_size=10,
                               scales={'x': polyreg1_x_sc, 'y': polyreg1_y_sc})

    # Linear Regression

    w1 = 1
    polyreg1_vector_plane = plt.Lines(x=polyreg1_x,
                                      y=polyreg1_reg,
                                      colors=['red'],
                                      scales={'x': polyreg1_x_sc, 'y': polyreg1_y_sc})

    polyreg1_f = plt.Figure(marks=[polyreg1_bar, polyreg1_vector_plane],
                            axes=[polyreg1_ax_x, polyreg1_ax_y],
                            title='Quadratic Regression',
                            legend_location='bottom-right',
                            animation_duration=0)

    polyreg1_f.layout.height = '400px'
    polyreg1_f.layout.width = '400px'

    # Widgets

    polyreg1_beta_0 = widgets.FloatSlider(
        min=-4, max=4, step=0.1, value=polyreg1_beta_0, description=markdown.markdown(r"$\beta_0$"))
    polyreg1_beta_1 = widgets.FloatSlider(
        min=-2, max=2, step=0.1, value=polyreg1_beta_1, description=markdown.markdown(r"$\beta_1$"))
    polyreg1_beta_2 = widgets.FloatSlider(
        min=-2, max=2, step=0.1, value=polyreg1_beta_2, description=markdown.markdown(r"$\beta_2$"))

    # Fonction qui va interagir avec les widgets

    polyreg1_beta_0.style = {'description_width': '20px', 'width': '80%'}
    polyreg1_beta_1.style = {'description_width': '20px', 'width': '80%'}
    polyreg1_beta_2.style = {'description_width': '20px', 'width': '80%'}

    def polyreg1_update_regression(args):
        polyreg1_vector_plane.y = polyreg1_beta_0.value \
                                + polyreg1_beta_1.value * polyreg1_x \
                                + polyreg1_beta_2.value * (polyreg1_x**2)



    polyreg1_beta_0.observe(polyreg1_update_regression)
    polyreg1_beta_1.observe(polyreg1_update_regression)
    polyreg1_beta_2.observe(polyreg1_update_regression)

    polyreg1_sliders = widgets.VBox([polyreg1_beta_0, polyreg1_beta_1, polyreg1_beta_2])
    polyreg1_widget = widgets.HBox([polyreg1_f, polyreg1_sliders])
    polyreg1_widget.layout.align_items = 'center'
    display(polyreg1_widget)


def polynomial_regression2():
    ################################################################################

    # Simulation of Data

    polyreg2_n_samples = 30
    polyreg2_alpha_1 = 1
    polyreg2_alpha_2 = - 0.1
    polyreg2_bias = 1.5
    polyreg2_x_min = 0
    polyreg2_x_max = 10
    polyreg2_noise_var = 1

    polyreg2_x = np.linspace(polyreg2_x_min, polyreg2_x_max,polyreg2_n_samples)
    polyreg2_data = np.vstack([polyreg2_x,
                               polyreg2_alpha_2 * (polyreg2_x**2)\
                               + polyreg2_alpha_1 * polyreg2_x   \
                               + polyreg2_bias                   \
                               + np.random.normal(0, polyreg2_noise_var, polyreg2_n_samples)]).T


    # Regression parameters

    polyreg2_beta_0 = 0
    polyreg2_beta_1 = 0.5
    polyreg2_beta_2 = 1

    polyreg2_reg = polyreg2_beta_0 + polyreg2_beta_1 * polyreg2_x + polyreg2_beta_2 * (polyreg2_x**2)


    polyreg2_colors = ["blue"] * polyreg2_n_samples

    # Scales


    polyreg2_x_sc = plt.LinearScale(min=polyreg2_x_min, max=polyreg2_x_max)
    polyreg2_y_sc = plt.LinearScale(min=polyreg2_x_min, max=polyreg2_x_max)

    polyreg2_ax_x = plt.Axis(scale=polyreg2_x_sc,
                             grid_lines='none',
                             label='x')

    polyreg2_ax_y = plt.Axis(scale=polyreg2_y_sc,
                             orientation='vertical',
                             grid_lines='none',
                             label='y')

    # Scatter plot of data

    polyreg2_bar = plt.Scatter(x=polyreg2_data[:, 0],
                               y=polyreg2_data[:, 1],
                               colors=polyreg2_colors,
                               default_size=10,
                               scales={'x': polyreg2_x_sc, 'y': polyreg2_y_sc})

    # Linear Regression

    w1 = 1
    polyreg2_vector_plane = plt.Lines(x=polyreg2_x,
                                      y=polyreg2_reg,
                                      colors=['red'],
                                      scales={'x': polyreg2_x_sc, 'y': polyreg2_y_sc})

    polyreg2_f = plt.Figure(marks=[polyreg2_bar, polyreg2_vector_plane],
                            axes=[polyreg2_ax_x, polyreg2_ax_y],
                            title='Polynomial Regression',
                            legend_location='bottom-right',
                            animation_duration=0)

    polyreg2_f.layout.height = '400px'
    polyreg2_f.layout.width = '400px'

    # Widgets



    # Fonction qui va interagir avec les widgets

    polyreg2_widget = widgets.HBox([polyreg2_f])
    polyreg2_widget.layout.align_items = 'center'
    display(polyreg2_widget)

    @widgets.interact(d = widgets.IntSlider(min=1, max=20, value=1, description=markdown.markdown(r"$d$")))
    def polyreg2_update_regression(d):
        data = polyreg2_x.copy().reshape(polyreg2_n_samples,1)
        for i in range(2, d+1):
            data = np.hstack([data, (polyreg2_x**i).reshape(polyreg2_n_samples, 1)])
        lr = LinearRegression()
        lr.fit(data, polyreg2_data[:, 1])

        polyreg2_vector_plane.y = lr.predict(data)