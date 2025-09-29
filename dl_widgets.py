#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import ipywidgets as widgets
import bqplot as bq
import bqplot.pyplot as plt
from IPython.display import display, Markdown
import markdown
import time
from sklearn import datasets
import pandas as pd
import json as json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, hinge_loss, mean_squared_error

import random
from scipy.interpolate import interp1d
from sklearn.utils import shuffle

def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def enable_colab_widgets():
    if in_colab():
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
        except Exception:
            display(HTML("<b style='color:red'>⚠️ Attention :</b> "
                         "Les widgets interactifs peuvent ne pas fonctionner correctement dans Colab. "
                         "Essaie plutôt dans Jupyter Notebook/Lab."))

enable_colab_widgets()

def show_losses():
    iris_X = datasets.load_iris()['data'][:,:2]
    iris_y = datasets.load_iris()['target']
    iris_y[iris_y == 2] = 1

    iris_y = 1 - iris_y

    colors = []
    for i in range(iris_y.shape[0]):
        if (iris_y[i] == 0):
            colors.append('green')
        else:
            colors.append('orange')

    scaled = iris_X[:,:2] - np.array([np.mean(iris_X[:,0]), np.mean(iris_X[:,1])])
    scaled = scaled / np.array([np.std(iris_X[:,0]), np.std(iris_X[:,1])])


    def model_definition(X, y):
        lr = LogisticRegression(fit_intercept=False)
        return lr.fit(X, y)


    def change_angle(model, alpha):
        alpha = alpha*np.pi/180
        x = -np.sin(alpha)
        y = np.cos(alpha)
        lr.coef_[0,0]=x
        lr.coef_[0,1]=y
        return model


    def loss(model, X, y):
        y_pred = model.predict_proba(X)
        return log_loss(y, y_pred)

    def loss_angular(model, X, y, alpha):
        model = change_angle(model, alpha)
        return loss(model, X, y)

    def metric_mse(model, X, y, alpha):
        model = change_angle(model, alpha)
        y_pred = model.predict_proba(X)[:,1]
        return mean_squared_error(y, y_pred)

    def metric_hinge(model, X, y, alpha):
        model = change_angle(model, alpha)
        y_pred = model.decision_function(X)
        return hinge_loss(y*2-1, y_pred)


    lr = model_definition(scaled, iris_y)


    init_beta = 0


    betas = np.linspace(-90, 120, 500)

    data_loss = [loss_angular(lr, scaled, iris_y, b) for b in betas]

    data_mse = [metric_mse(lr, scaled, iris_y, b) for b in betas]

    data_hinge = [metric_hinge(lr, scaled, iris_y, b) for b in betas]



    ################################################################################    

                                # Scales

    from IPython.display import display

    sep2_x_sc = plt.LinearScale(min = -4, max = 4)
    sep2_y_sc = plt.LinearScale(min = -4, max = 4)

    linreg2_loss_x_sc = plt.LinearScale(min = -90, max = 120)
    linreg2_loss_y_sc = plt.LinearScale(min = 0, max = 1.5)

    mse_x_sc = plt.LinearScale(min = -90, max = 120)
    mse_y_sc = plt.LinearScale(min = 0, max = 1)

    hinge_x_sc = plt.LinearScale(min = -90, max = 120)
    hinge_y_sc = plt.LinearScale(min = 0, max = 1.5)

    linreg2_loss_ax_x = plt.Axis(scale=linreg2_loss_x_sc,
                                 grid_lines='none',
                                 label='\\u03B2')

    linreg2_loss_ax_y = plt.Axis(scale=linreg2_loss_y_sc,
                                 orientation='vertical',
                                 grid_lines='none',
                                 label='Loss(\\u03B2)')

    mse_ax_x = plt.Axis(scale=mse_x_sc,
                                 grid_lines='none',
                                 label='\\u03B2')

    mse_ax_y = plt.Axis(scale=mse_y_sc,
                                 orientation='vertical',
                                 grid_lines='none',
                                 label='ACC(\\u03B2)')

    hinge_ax_x = plt.Axis(scale=hinge_x_sc,
                                 grid_lines='none',
                                 label='\\u03B2')

    hinge_ax_y = plt.Axis(scale=hinge_y_sc,
                                 orientation='vertical',
                                 grid_lines='none',
                                 label='ACC(\\u03B2)')



    sep2_ax_x = plt.Axis(scale=sep2_x_sc,
                    grid_lines='solid',
                    label='Sepal Length')

    sep2_ax_y = plt.Axis(scale=sep2_y_sc,
                    orientation='vertical',
                    grid_lines='solid',
                    label='Sepal Width')

                              # Scatter plot

    linreg2_loss_dot = plt.Scatter(x = [init_beta],
                                   y = [loss_angular(lr, scaled, iris_y, init_beta)],
                                   colors = ['red'],
                                   default_size = 100,
                                   scales={'x': linreg2_loss_x_sc, 'y': linreg2_loss_y_sc})


    linreg2_loss = plt.Lines(x = betas,
                             y = data_loss,
                             colors = ['green'],
                             default_size = 10,
                             scales={'x': linreg2_loss_x_sc, 'y': linreg2_loss_y_sc})



    ## ACCCCCCC


    mse_dot = plt.Scatter(x = [init_beta],
                                   y = [metric_mse(lr, scaled, iris_y, init_beta)],
                                   colors = ['red'],
                                   default_size = 100,
                                   scales={'x': mse_x_sc, 'y': mse_y_sc})


    mse_line = plt.Lines(x = betas,
                             y = data_mse,
                             colors = ['green'],
                             default_size = 10,
                             scales={'x': mse_x_sc, 'y': mse_y_sc})


    hinge_dot = plt.Scatter(x = [init_beta],
                                   y = [metric_hinge(lr, scaled, iris_y, init_beta)],
                                   colors = ['red'],
                                   default_size = 100,
                                   scales={'x': hinge_x_sc, 'y': hinge_y_sc})


    hinge_line = plt.Lines(x = betas,
                             y = data_hinge,
                             colors = ['green'],
                             default_size = 10,
                             scales={'x': hinge_x_sc, 'y': hinge_y_sc})




    ## DATA

    sep2_bar = plt.Scatter(x = scaled[:,0],
                      y = scaled[:,1]-1,
                      colors = colors,
                      default_size = 10,
                      scales={'x': sep2_x_sc, 'y': sep2_y_sc})

                                 # Vector

    w1, w2 = 0, 1
    w = np.array([w1, w2])

    sep2_vector_line = plt.Lines(x = np.array([0, w1]),
                            y = np.array([0, w2]),
                            colors = ['red', 'red'],
                            scales={'x': sep2_x_sc, 'y': sep2_y_sc})

    # sep2_vector_label = plt.Label(x = [w1],
    #                          y = [w2],
    #                          text = ['(w1, w2)'],
    #                          size = [10])

    sep2_vector_plane = plt.Lines(x = [-30*(w2 / np.linalg.norm(w)), 30*(w2 / np.linalg.norm(w))],
                             y = [30*(w1 / np.linalg.norm(w)), -30*(w1 / np.linalg.norm(w))],
                             colors = ['red', 'red'],
                             scales={'x': sep2_x_sc, 'y': sep2_y_sc})

    sep2_f = plt.Figure(marks=[sep2_bar, sep2_vector_line, sep2_vector_plane],
                   axes=[sep2_ax_x, sep2_ax_y],
                   title='Iris Dataset',
                   legend_location='bottom-right')

    linreg2_loss_f = plt.Figure(marks = [linreg2_loss, linreg2_loss_dot],
                                axes = [linreg2_loss_ax_x, linreg2_loss_ax_y],
                                title = "Loss: cross-entropy",
                                animation_duration = 0)


    mse_f = plt.Figure(marks = [mse_line, mse_dot],
                                axes = [mse_ax_x, mse_ax_y],
                                title = "Mean Square",
                                animation_duration = 0)


    hinge_f = plt.Figure(marks = [hinge_line, hinge_dot],
                                axes = [hinge_ax_x, hinge_ax_y],
                                title = "Hinge",
                                animation_duration = 0)

    sep2_f.layout.height = '400px'
    sep2_f.layout.width = '400px'

    linreg2_loss_f.layout.height = '400px'
    linreg2_loss_f.layout.width = '400px'

    mse_f.layout.height = '400px'
    mse_f.layout.width = '400px'

    hinge_f.layout.height = '400px'
    hinge_f.layout.width = '400px'




    # progress.value += 1
    linreg2_beta_hat = widgets.FloatSlider(min=-90, max=120, step=1, value=0, description=markdown.markdown(r"$\\beta$"))
    linreg2_beta_hat.style = {'description_width': '20px', 'width' : '80%'}


    def linreg2_update_loss(args):
        linreg2_loss_dot.x = [linreg2_beta_hat.value]
        linreg2_loss_dot.y = [loss_angular(lr, scaled, iris_y, linreg2_beta_hat.value)]

    def linreg2_update_mse(args):
        mse_dot.x = [linreg2_beta_hat.value]
        mse_dot.y = [metric_mse(lr, scaled, iris_y, linreg2_beta_hat.value)]

    def linreg2_update_hinge(args):
        hinge_dot.x = [linreg2_beta_hat.value]
        hinge_dot.y = [metric_hinge(lr, scaled, iris_y, linreg2_beta_hat.value)]

    def linreg2_update_regression(args):
        alpha = linreg2_beta_hat.value
        alpha = alpha*np.pi/180
        w1 = -np.sin(alpha)
        w2 = np.cos(alpha)
        sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
        sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]
        w = np.array([w1, w2])
        sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
        sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]



    linreg2_beta_hat.observe(linreg2_update_loss)
    linreg2_beta_hat.observe(linreg2_update_mse)
    linreg2_beta_hat.observe(linreg2_update_hinge)
    linreg2_beta_hat.observe(linreg2_update_regression)


    plots = widgets.HBox([sep2_f, linreg2_loss_f])
    widget = widgets.VBox([plots, linreg2_beta_hat, widgets.HBox([mse_f, hinge_f])])

    widget.layout.align_items = 'center'
    display(widget)


    sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
    sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]

    sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
    sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]

    def h(alpha):
    #     print(alpha)
        alpha = alpha*np.pi/180
        w1 = -np.sin(alpha)
        w2 = np.cos(alpha)
        sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
        sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]
        w = np.array([w1, w2])
        sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
        sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]

##################################################################################################################        
        
def show_accuracy():
    iris_X = datasets.load_iris()['data'][:,:2]
    iris_y = datasets.load_iris()['target']
    iris_y[iris_y == 2] = 1

    iris_y = 1 - iris_y

    colors = []
    for i in range(iris_y.shape[0]):
        if (iris_y[i] == 0):
            colors.append('green')
        else:
            colors.append('orange')

    scaled = iris_X[:,:2] - np.array([np.mean(iris_X[:,0]), np.mean(iris_X[:,1])])
    scaled = scaled / np.array([np.std(iris_X[:,0]), np.std(iris_X[:,1])])


    def model_definition(X, y):
        lr = LogisticRegression(fit_intercept=False)
        return lr.fit(X, y)


    def change_angle(model, alpha):
        alpha = alpha*np.pi/180
        x = -np.sin(alpha)
        y = np.cos(alpha)
        lr.coef_[0,0]=x
        lr.coef_[0,1]=y
        return model


    def loss(model, X, y):
        y_pred = model.predict_proba(X)
        return log_loss(y, y_pred)

    def loss_angular(model, X, y, alpha):
        model = change_angle(model, alpha)
        return loss(model, X, y)

    def metric_acc(model, X, y, alpha):
        model = change_angle(model, alpha)
        return model.score(X, y)

    lr = model_definition(scaled, iris_y)


    init_beta = 0


    betas = np.linspace(-90, 120, 500)

    data_loss = [loss_angular(lr, scaled, iris_y, b) for b in betas]

    data_acc = [metric_acc(lr, scaled, iris_y, b) for b in betas]


    ################################################################################    

                                # Scales

    from IPython.display import display

    sep2_x_sc = plt.LinearScale(min = -4, max = 4)
    sep2_y_sc = plt.LinearScale(min = -4, max = 4)

    linreg2_loss_x_sc = plt.LinearScale(min = -90, max = 120)
    linreg2_loss_y_sc = plt.LinearScale(min = 0, max = 1.5)

    acc_x_sc = plt.LinearScale(min = -90, max = 120)
    acc_y_sc = plt.LinearScale(min = 0, max = 1)

    linreg2_loss_ax_x = plt.Axis(scale=linreg2_loss_x_sc,
                                 grid_lines='none',
                                 label='\\u03B2')

    linreg2_loss_ax_y = plt.Axis(scale=linreg2_loss_y_sc,
                                 orientation='vertical',
                                 grid_lines='none',
                                 label='Loss(\\u03B2)')

    acc_ax_x = plt.Axis(scale=acc_x_sc,
                                 grid_lines='none',
                                 label='\\u03B2')

    acc_ax_y = plt.Axis(scale=acc_y_sc,
                                 orientation='vertical',
                                 grid_lines='none',
                                 label='ACC(\\u03B2)')



    sep2_ax_x = plt.Axis(scale=sep2_x_sc,
                    grid_lines='solid',
                    label='Sepal Length')

    sep2_ax_y = plt.Axis(scale=sep2_y_sc,
                    orientation='vertical',
                    grid_lines='solid',
                    label='Sepal Width')

                              # Scatter plot

    linreg2_loss_dot = plt.Scatter(x = [init_beta],
                                   y = [loss_angular(lr, scaled, iris_y, init_beta)],
                                   colors = ['red'],
                                   default_size = 100,
                                   scales={'x': linreg2_loss_x_sc, 'y': linreg2_loss_y_sc})


    linreg2_loss = plt.Lines(x = betas,
                             y = data_loss,
                             colors = ['green'],
                             default_size = 10,
                             scales={'x': linreg2_loss_x_sc, 'y': linreg2_loss_y_sc})



    ## ACCCCCCC


    acc_dot = plt.Scatter(x = [init_beta],
                                   y = [metric_acc(lr, scaled, iris_y, init_beta)],
                                   colors = ['red'],
                                   default_size = 100,
                                   scales={'x': acc_x_sc, 'y': acc_y_sc})


    acc_line = plt.Lines(x = betas,
                             y = data_acc,
                             colors = ['green'],
                             default_size = 10,
                             scales={'x': acc_x_sc, 'y': acc_y_sc})




    ## DATA

    sep2_bar = plt.Scatter(x = scaled[:,0],
                      y = scaled[:,1]-1,
                      colors = colors,
                      default_size = 10,
                      scales={'x': sep2_x_sc, 'y': sep2_y_sc})

                                 # Vector

    w1, w2 = 0, 1
    w = np.array([w1, w2])

    sep2_vector_line = plt.Lines(x = np.array([0, w1]),
                            y = np.array([0, w2]),
                            colors = ['red', 'red'],
                            scales={'x': sep2_x_sc, 'y': sep2_y_sc})

    # sep2_vector_label = plt.Label(x = [w1],
    #                          y = [w2],
    #                          text = ['(w1, w2)'],
    #                          size = [10])

    sep2_vector_plane = plt.Lines(x = [-30*(w2 / np.linalg.norm(w)), 30*(w2 / np.linalg.norm(w))],
                             y = [30*(w1 / np.linalg.norm(w)), -30*(w1 / np.linalg.norm(w))],
                             colors = ['red', 'red'],
                             scales={'x': sep2_x_sc, 'y': sep2_y_sc})

    sep2_f = plt.Figure(marks=[sep2_bar, sep2_vector_line, sep2_vector_plane],
                   axes=[sep2_ax_x, sep2_ax_y],
                   title='Iris Dataset',
                   legend_location='bottom-right')

    linreg2_loss_f = plt.Figure(marks = [linreg2_loss, linreg2_loss_dot],
                                axes = [linreg2_loss_ax_x, linreg2_loss_ax_y],
                                title = "Loss: cross-entropy",
                                animation_duration = 0)


    acc_f = plt.Figure(marks = [acc_line, acc_dot],
                                axes = [acc_ax_x, acc_ax_y],
                                title = "Metric: Accuracy",
                                animation_duration = 0)

    sep2_f.layout.height = '400px'
    sep2_f.layout.width = '400px'

    linreg2_loss_f.layout.height = '400px'
    linreg2_loss_f.layout.width = '400px'

    acc_f.layout.height = '400px'
    acc_f.layout.width = '800px'




    # progress.value += 1
    linreg2_beta_hat = widgets.FloatSlider(min=-90, max=120, step=1, value=0, description=markdown.markdown(r"$\\beta$"))
    linreg2_beta_hat.style = {'description_width': '20px', 'width' : '80%'}


    def linreg2_update_loss(args):
        linreg2_loss_dot.x = [linreg2_beta_hat.value]
        linreg2_loss_dot.y = [loss_angular(lr, scaled, iris_y, linreg2_beta_hat.value)]

    def linreg2_update_acc(args):
        acc_dot.x = [linreg2_beta_hat.value]
        acc_dot.y = [metric_acc(lr, scaled, iris_y, linreg2_beta_hat.value)]

    def linreg2_update_regression(args):
        alpha = linreg2_beta_hat.value
        alpha = alpha*np.pi/180
        w1 = -np.sin(alpha)
        w2 = np.cos(alpha)
        sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
        sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]
        w = np.array([w1, w2])
        sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
        sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]



    linreg2_beta_hat.observe(linreg2_update_loss)
    linreg2_beta_hat.observe(linreg2_update_acc)
    linreg2_beta_hat.observe(linreg2_update_regression)


    plots = widgets.HBox([sep2_f, linreg2_loss_f])
    widget = widgets.VBox([plots, linreg2_beta_hat, acc_f])

    widget.layout.align_items = 'center'
    display(widget)


    sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
    sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]

    sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
    sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]

    def h(alpha):
    #     print(alpha)
        alpha = alpha*np.pi/180
        w1 = -np.sin(alpha)
        w2 = np.cos(alpha)
        sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
        sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]
        w = np.array([w1, w2])
        sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
        sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]

#####################################################################################################################        
        
def show_batch_gradient_descent():
    iris_X = datasets.load_iris()['data'][:,:2]
    iris_y = datasets.load_iris()['target']
    iris_y[iris_y == 2] = 1

    iris_y = 1 - iris_y



    scaled = iris_X[:,:2] - np.array([np.mean(iris_X[:,0]), np.mean(iris_X[:,1])])
    scaled = scaled / np.array([np.std(iris_X[:,0]), np.std(iris_X[:,1])])

    scaled, iris_y = shuffle(scaled, iris_y, random_state=1234)


    def color(y):
        colors = []
        for i in range(y.shape[0]):
            if (y[i] == 0):
                colors.append('green')
            else:
                colors.append('orange')
        return colors

    colors = color(iris_y)


    def split(X, batch_size): 
        return [X[i:i+batch_size] for i in range(0, batch_size*(int(len(X)/batch_size)), batch_size)]




    def model_definition(X, y):
        lr = LogisticRegression(fit_intercept=False)
        return lr.fit(X, y)


    def change_angle(model, alpha):
        alpha = alpha*np.pi/180
        x = -np.sin(alpha)
        y = np.cos(alpha)
        lr.coef_[0,0]=x
        lr.coef_[0,1]=y
        return model

    def select_batch():
        i = random.randint(0,len(X_splits)-1)
        return X_splits[i], y_splits[i], data_batch_loss[i], loss_batch_linear[i], i

    def loss(model, X, y):
        y_pred = model.predict_proba(X)
        return log_loss(y, y_pred, labels=[0,1])


    def gradloss(model, X, y, alpha, dt=0.1):
        model = change_angle(model, alpha-dt)
        loss1 = loss(model, X, y)
        model = change_angle(model, alpha+dt)
        loss2 = loss(model, X, y)
        return (loss2-loss1)/(2*dt)

    def loss_angular(model, X, y, alpha):
        model = change_angle(model, alpha)
        return loss(model, X, y)

    def metric_acc(model, X, y, alpha):
        model = change_angle(model, alpha)
        return model.score(X, y)

    def metric_mse(model, X, y, alpha):
        model = change_angle(model, alpha)
        y_pred = model.predict_proba(X)[:,1]
        return mean_squared_error(y, y_pred)

    def metric_hinge(model, X, y, alpha):
        model = change_angle(model, alpha)
        y_pred = model.decision_function(X)
        return hinge_loss(y*2-1, y_pred)

    lr = model_definition(scaled, iris_y)


    init_beta = 0

    beta_min = -40

    beta_max = 480

    batch_size = 10

    betas = np.linspace(-90, beta_max, 100)

    data_loss_global = [loss_angular(lr, scaled, iris_y, b) for b in betas]

    data_acc = [metric_acc(lr, scaled, iris_y, b) for b in betas]


    loss_global_linear = interp1d(betas, data_loss_global, kind='linear', bounds_error=False, fill_value='extrapolate')




    X_splits = split(scaled, batch_size)

    y_splits = split(iris_y, batch_size)


    data_batch_loss = [[loss_angular(lr, X_b, y_b, b) for b in betas] for X_b, y_b in zip(X_splits, y_splits)]

    loss_batch_linear = [interp1d(betas, X_loss, kind='linear', bounds_error=False, fill_value='extrapolate') for X_loss in data_batch_loss]

    loss_linear = loss_batch_linear[0]

    def gradlossApprox(alpha, dt=1):
        return (loss_linear(alpha+dt)-loss_linear(alpha-dt))/(2*dt)
    ################################################################################    

                                # Scales

    from IPython.display import display

    sep2_x_sc = plt.LinearScale(min = -4, max = 4)
    sep2_y_sc = plt.LinearScale(min = -4, max = 4)

    linreg2_loss_x_sc = plt.LinearScale(min = beta_min, max = beta_max)
    linreg2_loss_y_sc = plt.LinearScale(min = 0, max = 1.5)

    acc_x_sc = plt.LinearScale(min = beta_min, max = beta_max)
    acc_y_sc = plt.LinearScale(min = 0, max = 1)

    linreg2_loss_ax_x = plt.Axis(scale=linreg2_loss_x_sc,
                                 grid_lines='none',
                                 label='\\u03B2')

    linreg2_loss_ax_y = plt.Axis(scale=linreg2_loss_y_sc,
                                 orientation='vertical',
                                 grid_lines='none',
                                 label='Loss(\\u03B2)')

    acc_ax_x = plt.Axis(scale=acc_x_sc,
                                 grid_lines='none',
                                 label='\\u03B2')

    acc_ax_y = plt.Axis(scale=acc_y_sc,
                                 orientation='vertical',
                                 grid_lines='none',
                                 label='ACC(\\u03B2)')



    sep2_ax_x = plt.Axis(scale=sep2_x_sc,
                    grid_lines='solid',
                    label='Sepal Length')

    sep2_ax_y = plt.Axis(scale=sep2_y_sc,
                    orientation='vertical',
                    grid_lines='solid',
                    label='Sepal Width')

                              # Scatter plot

    # linreg2_loss_dot = plt.Scatter(x = [init_beta],
    #                                y = [loss_angular(lr, scaled, iris_y, init_beta)],
    #                                colors = ['red'],
    #                                default_size = 100,
    #                                scales={'x': linreg2_loss_x_sc, 'y': linreg2_loss_y_sc})


    linreg2_loss = plt.Lines(x = betas,
                             y = data_loss_global,
                             colors = ['green'],
                             default_size = 10,
                             scales={'x': linreg2_loss_x_sc, 'y': linreg2_loss_y_sc})

    linreg2_loss_local = plt.Lines(x = betas,
                             y = data_batch_loss[0],
                             colors = ['blue'],
                             opacities=[0.7,0.7],
                             default_size = 10,
                             scales={'x': linreg2_loss_x_sc, 'y': linreg2_loss_y_sc})

    xk = np.array([init_beta])

    yk = loss_global_linear(xk)

    grad2_point = plt.Scatter(x = xk,
                        y = yk,
                        colors = ["red"],
                        scales = {'x': linreg2_loss_x_sc, 'y' : linreg2_loss_y_sc})

    grad2_point_lines = plt.Lines(x = loss_global_linear(xk),
                            y = loss_global_linear(xk),
                            colors = ["red"],
                            scales = {'x': linreg2_loss_x_sc, 'y' : linreg2_loss_y_sc})



    grad2_label_f_prim = plt.label([" "],
                        x = xk,
                        y = loss_global_linear(xk),
                        x_offset = 0,
                        y_offset = 0,
                        default_size = 20,
                        font_weight = 'bolder',
                        update_on_move = True,
                        colors = ["red"])


    label_batch_loss = plt.label(["Loss Batch"],
                        x = [0],
                        y = [20],
                        x_offset = 0,
                        y_offset = 0,
                        opacities=[0.7,0.7],
                        default_size = 15,
                        font_weight = 'bolder',
                        update_on_move = True,
                        colors = ["blue"])

    label_global_loss = plt.label(["Loss all dataset"],
                        x = [0],
                        y = [5],
                        x_offset = 0,
                        y_offset = 0,
                        default_size = 15,
                        font_weight = 'bolder',
                        update_on_move = True,
                        colors = ["Green"])


    grad2_lines_x0 = plt.Lines(x = [init_beta, init_beta],
                         y = [0, loss_global_linear(init_beta)],
                         scales = {'x': linreg2_loss_x_sc, 'y' : linreg2_loss_y_sc},
                         line_style = "dashed",
                         colors = ["red"])

    ## ACCCCCCC


    acc_dot = plt.Scatter(x = [init_beta],
                                   y = [metric_acc(lr, scaled, iris_y, init_beta)],
                                   colors = ['red'],
                                   default_size = 100,
                                   scales={'x': acc_x_sc, 'y': acc_y_sc})


    acc_line = plt.Lines(x = betas,
                             y = data_acc,
                             colors = ['green'],
                             default_size = 10,
                             scales={'x': acc_x_sc, 'y': acc_y_sc})




    ## DATA

    sep2_bar = plt.Scatter(x = scaled[:,0],
                      y = scaled[:,1]-1,
                      colors = colors,
                      default_size = 10,
                      opacities=[0.05,0.05],
                      scales={'x': sep2_x_sc, 'y': sep2_y_sc})
    # from bqplot import Tooltip
    # sep2_bar.tooltip = Tooltip(fields=["x", "y"], labels=["Wine Category", "Avg Ash/Flavanoids"])


    label_batch_id = plt.label(["Batch 0"],
                        x = [0],
                        y = [160],
                        x_offset = 0,
                        y_offset = 0,
                        default_size = 25,
                        font_weight = 'bolder',
                        update_on_move = True,
                        colors = ["black"])


    batch_ids = [0]

    sep2_batch = plt.Scatter(x = X_splits[batch_ids[0]][:,0],
                      y = X_splits[batch_ids[0]][:,1]-1,
                      colors = color(y_splits[batch_ids[0]]),
                      default_size = 10,
                      opacities=[1,1],
                      scales={'x': sep2_x_sc, 'y': sep2_y_sc})

                                 # Vector

    w1, w2 = 0, 1
    w = np.array([w1, w2])

    sep2_vector_line = plt.Lines(x = np.array([0, w1]),
                            y = np.array([0, w2]),
                            colors = ['red', 'red'],
                            scales={'x': sep2_x_sc, 'y': sep2_y_sc})


    # sep2_vector_label = plt.Label(x = [w1],
    #                          y = [w2],
    #                          text = ['(w1, w2)'],
    #                          size = [10])

    sep2_vector_plane = plt.Lines(x = [-30*(w2 / np.linalg.norm(w)), 30*(w2 / np.linalg.norm(w))],
                             y = [30*(w1 / np.linalg.norm(w)), -30*(w1 / np.linalg.norm(w))],
                             colors = ['red', 'red'],
                             scales={'x': sep2_x_sc, 'y': sep2_y_sc})

    sep2_f = plt.Figure(marks=[sep2_bar, sep2_vector_line, sep2_vector_plane, sep2_batch, label_batch_id],
                   axes=[sep2_ax_x, sep2_ax_y],
                   title='Batch of Iris Dataset',
                   legend_location='bottom-right')

    linreg2_loss_f = plt.Figure(marks = [linreg2_loss, grad2_point, grad2_point_lines, grad2_lines_x0, grad2_label_f_prim, linreg2_loss_local, label_batch_loss, label_global_loss],
                                axes = [linreg2_loss_ax_x, linreg2_loss_ax_y],
                                title = "Loss",
                                animation_duration = 0)


    acc_f = plt.Figure(marks = [acc_line, acc_dot],
                                axes = [acc_ax_x, acc_ax_y],
                                title = "Metric: Accuracy",
                                animation_duration = 0)

    sep2_f.layout.height = '400px'
    sep2_f.layout.width = '400px'

    linreg2_loss_f.layout.height = '400px'
    linreg2_loss_f.layout.width = '400px'

    acc_f.layout.height = '400px'
    acc_f.layout.width = '800px'




    # progress.value += 1
    linreg2_beta_hat = widgets.FloatSlider(min=beta_min, max=beta_max, step=1, value=0, description=markdown.markdown(r"$\\beta_0$"))

    grad2_learning_rate = widgets.BoundedFloatText(min=0.001, max=5, step=0.01, value = 0.5, description = "Learning rate")
    grad2_etape_play = widgets.Play(value = 0, interval = 250, min=0, max=30, step=1, disabled=False)
    grad2_etape = widgets.IntSlider(value = 0, min = 0, max = 30, step = 1, description = "Step")

    batch_size_widget = widgets.IntSlider(value = batch_size, min = 2, max = 50, step = 1, description = "Batch size")

    grad2_hbox = widgets.HBox([grad2_etape_play,grad2_etape])
    widgets.jslink((grad2_etape_play, 'value'), (grad2_etape, 'value'))

    select_loss = widgets.Dropdown(
        options=[('Cross Entropy', 0), ('Hinge', 1), ('Accuracy', 2)],
        value=0,
        description='Loss:',
    )

    linreg2_beta_hat.style = {'description_width': '20px', 'width' : '80%'}


    # def linreg2_update_loss(args):
    #     linreg2_loss_dot.x = [linreg2_beta_hat.value]
    #     linreg2_loss_dot.y = [loss_angular(lr, scaled, iris_y, linreg2_beta_hat.value)]

    def linreg2_update_acc(args):
        acc_dot.x = [linreg2_beta_hat.value]
        acc_dot.y = [metric_acc(lr, scaled, iris_y, linreg2_beta_hat.value)]

    def linreg2_update_regression(args):
        alpha = linreg2_beta_hat.value
        alpha = alpha*np.pi/180
        w1 = -np.sin(alpha)
        w2 = np.cos(alpha)
        sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
        sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]
        w = np.array([w1, w2])
        sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
        sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]


    def update_regression(alpha):
        alpha = alpha*np.pi/180
        w1 = -np.sin(alpha)
        w2 = np.cos(alpha)
        sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
        sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]
        w = np.array([w1, w2])
        sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
        sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]


    # linreg2_beta_hat.observe(linreg2_update_loss)
    # linreg2_beta_hat.observe(linreg2_update_acc)
    linreg2_beta_hat.observe(linreg2_update_regression)


    plots = widgets.HBox([sep2_f, linreg2_loss_f])
    widget = widgets.VBox([plots, linreg2_beta_hat, grad2_learning_rate, select_loss, batch_size_widget, grad2_etape, grad2_etape_play])

    widget.layout.align_items = 'center'
    display(widget)


    sep2_vector_line.x = [0, w1, 0.8*w1 - w2/10, w1, 0.8*w1 + w2/10]
    sep2_vector_line.y = [0, w2, 0.8*w2 + w1/10, w2, 0.8*w2 - w1/10]

    sep2_vector_plane.x = [-30*w2 / np.linalg.norm(w), 30*w2 / np.linalg.norm(w)]
    sep2_vector_plane.y = [30*w1 / np.linalg.norm(w), -30*w1 / np.linalg.norm(w)]

    def compute_gradient(args):
        global xk, yk, batch_ids
        xk = np.zeros(31)
        yk = np.zeros(31)
        xk[0] = linreg2_beta_hat.value
        yk[0] = loss_global_linear(xk[0])
        batch_ids = []
        global loss_linear
        for k in np.arange(30)+1:
            X_b, y_b, loss_b, lin_loss, batch_id = select_batch()
            loss_linear = lin_loss
            batch_ids.append(batch_id)
            xk[k] = xk[k-1] - grad2_learning_rate.value*5000*gradlossApprox(xk[k-1])
            yk[k] = loss_global_linear(xk[k])

    def grad2_gradient_plot(change):

        X_b, y_b = X_splits[batch_ids[grad2_etape.value]], y_splits[batch_ids[grad2_etape.value]]

        loss_b = data_batch_loss[batch_ids[grad2_etape.value]]

        show_data(X_b, y_b)
        show_loss(loss_b)
        grad2_point.x = xk[:grad2_etape.value+1]
        grad2_point.y = yk[:grad2_etape.value+1]

        grad2_point_lines.x = xk[:grad2_etape.value+1]
        grad2_point_lines.y = yk[:grad2_etape.value+1]

        update_regression(xk[grad2_etape.value])

    #     grad2_label_x0.x = [linreg2_beta_hat.value]
        grad2_lines_x0.x = [linreg2_beta_hat.value, linreg2_beta_hat.value]
        grad2_lines_x0.y = [0, loss_global_linear(linreg2_beta_hat.value)]

        grad2_label_f_prim.x = [-9]
        grad2_label_f_prim.y = [178]

        global loss_linear
        loss_linear = loss_global_linear
        grad2_label_f_prim.text = ["Step " + str(grad2_etape.value) + ", Lglobal'(𝛽"+str(grad2_etape.value)+") = "+str(np.round(500*gradlossApprox(xk[grad2_etape.value]), 3))]

        label_batch_id.text = ['Batch ' + str(batch_ids[grad2_etape.value])]
    def plot_dots(args):
        compute_gradient('')
        grad2_point.x = xk[:grad2_etape.value+1]
        grad2_point.y = yk[:grad2_etape.value+1]

        grad2_point_lines.x = xk[:grad2_etape.value+1]
        grad2_point_lines.y = yk[:grad2_etape.value+1]

        grad2_lines_x0.x = [linreg2_beta_hat.value, linreg2_beta_hat.value]
        grad2_lines_x0.y = [0, loss_global_linear(linreg2_beta_hat.value)]


    grad2_etape_play.observe(grad2_gradient_plot)
    linreg2_beta_hat.observe(plot_dots)
    # grad2_learning_rate.observe(grad2_gradient_plot)


    old_loss = 0

    def linreg2_update_loss(args):
        global old_loss
        grad2_etape.value = 0
        if old_loss == select_loss.value:
            return
        else :
            old_loss = select_loss.value
        grad2_etape_play._playing = False
        grad2_etape.value = 0
        beta_i = linreg2_beta_hat.value


        if select_loss.value==0:
            data_loss = [loss_angular(lr, scaled, iris_y, b) for b in betas]
        elif select_loss.value==1:
            data_loss = [metric_hinge(lr, scaled, iris_y, b) for b in betas]
        else :
            data_loss = [metric_acc(lr, scaled, iris_y, b) for b in betas]

        global loss_linear, data_batch_loss, loss_batch_linear, loss_global_linear


        loss_linear = interp1d(betas, data_loss, kind='linear', bounds_error=False, fill_value='extrapolate')
        loss_global_linear = loss_linear

        if select_loss.value==0:
            loss_function = loss_angular
        elif select_loss.value==1:
            loss_function = metric_hinge
        else :
            loss_function = metric_acc


        data_batch_loss = [[loss_function(lr, X_b, y_b, b) for b in betas] for X_b, y_b in zip(X_splits, y_splits)]

        loss_batch_linear = [interp1d(betas, X_loss, kind='linear', bounds_error=False, fill_value='extrapolate') for X_loss in data_batch_loss]



        linreg2_loss.y = data_loss
        linreg2_loss_local.y = data_batch_loss[0]
        compute_gradient('')
        plot_dots('')

    #     grad2_gradient_plot('')

    old_batch_size = batch_size

    def change_batch_size(args):

        global old_batch_size
        if old_batch_size == batch_size_widget.value:
            return
        else :
            old_batch_size = batch_size_widget.value

    #     grad2_etape_play._playing = False
        grad2_etape.value = 0
        batch_size = batch_size_widget.value
        global X_splits, y_splits, data_batch_loss, loss_batch_linear
        X_splits = split(scaled, batch_size)
        y_splits = split(iris_y, batch_size)

        if select_loss.value==0:
            loss_function = loss_angular
        elif select_loss.value==1:
            loss_function = metric_hinge
        else :
            loss_function = metric_acc


        data_batch_loss = [[loss_function(lr, X_b, y_b, b) for b in betas] for X_b, y_b in zip(X_splits, y_splits)]
        linreg2_loss_local.y = data_batch_loss[0]
        loss_batch_linear = [interp1d(betas, X_loss, kind='linear', bounds_error=False, fill_value='extrapolate') for X_loss in data_batch_loss]
        show_data(X_splits[0], y_splits[0])
        compute_gradient('')


    def show_data(X, y):
        sep2_batch.x, sep2_batch.y, sep2_batch.colors= X[:,0], X[:,1]-1, color(y)

    def show_loss(loss_d):
        linreg2_loss_local.y = loss_d

    batch_size_widget.observe(change_batch_size)
    select_loss.observe(linreg2_update_loss)

    compute_gradient('')
    grad2_learning_rate.observe(compute_gradient)

