import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
import numpy as np

import ipywidgets as widgets
from IPython.display import display, clear_output

import ipywidgets as widgets
from ipywidgets import Button, Layout, HTML, VBox, HBox


def show_log_regression():
    warnings.filterwarnings("ignore")
    def dataset(data='make_classification'):
        if data == 'make_classification':
            X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
        elif data == 'make_moons':
            X, y = make_moons(noise=0.3, random_state=0)

        elif data == 'make_circles':
            X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

        xx, yy = meshgrid(X, y, h=.1)

        return X, y, xx, yy



    def meshgrid(X, y, h=.1):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
        return xx, yy

    X, y, xx, yy = dataset(data='make_classification')

        # Plot the testing points

    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
    #                edgecolors='k')
    params = {'solver':"lbfgs", 'C': 1000}

    cm = plt.cm.RdBu

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])


    def fit_model(params, X, y, xx, yy):
        clf = LogisticRegression(solver=params['solver'], C=params['C'])
        clf.fit(X, y)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        return clf, Z

    # score = clf.score(X_test, y_test)

    def plot(params, X, y, xx, yy):
        contour_axis.clear()
        # just plot the dataset first


        contour_axis.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', cmap=cm_bright, label='Class 1',
                   edgecolors='k')
        contour_axis.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', cmap=cm_bright, label='Class 2',
                   edgecolors='k')

        contour_axis.set_xlabel('x1')
        contour_axis.set_ylabel('x2')
        contour_axis.legend()

        clf, Z = fit_model(params, X, y, xx, yy)


        # Put the result into a color plot

        decision_curve = contour_axis.contourf(xx, yy, Z, cmap=cm, alpha=.5)

        # plot decision boundary and margins
        margins = contour_axis.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.9,
               linestyles=['-'])

        # plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        #                 size=15, horizontalalignment='right')
    plt.ioff()
    contour_axis = plt.gca()

    # plot(params)


    import ipywidgets as widgets
    from ipywidgets import Button, Layout, HTML, VBox, HBox
    from IPython.display import display, clear_output
    # plt.subplot(121)

    h1 = HTML(value='<b>Datasets</b>')
    w0 = widgets.Dropdown(options=['make_classification', 'make_moons', 'make_circles'],
                            value='make_classification',
                            description='Dataset',
                            disabled = False)



    w1 = widgets.FloatText(description = 'C', value = 10.0)
    w2 = widgets.Dropdown(options=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' ],
                            value='lbfgs',
                            description='Solver',
                            disabled = False)

    h2 = HTML(value='<b>Hyperparameters</b>')

    box_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center')
    button = widgets.Button(description = 'Generate', layout = box_layout)


    grid1 = widgets.VBox(children=(h1, HTML(value=''), w0, HTML(value=''), HTML(value=''), HTML(value=''), h2, HTML(value=''), w1, w2))
    grid1.layout.align_items = 'center'
    grid2 = widgets.VBox(children=(grid1, HTML(value=''), HTML(value=''), HTML(value=''), button))
    grid2.layout.align_items = 'center'
    out=widgets.Output()

    box=widgets.HBox(children=(out, grid2))
    box.layout.align_items = 'center'

    display(box)

    def update_curve(a):
        # New fitting
        params['C'] = w1.value
        if w1.value<=0:
            params['C'] = 0.001
            w1.value = 0.001
        params['solver'] = w2.value
        X, y, xx, yy = dataset(w0.value)

        clf, Z = fit_model(params, X, y, xx, yy)
        # update curve
        plot(params, X, y, xx, yy)
        with out:
            clear_output(wait=True)
            display(contour_axis.figure)

    # plt.show()
    from IPython.display import display
    # widgets.HBox([kernel_grid, ])
    update_curve(None)

    button.on_click(update_curve)
    
    
    
    
    
    
    
    
    
    
    
    
def show_svm():
    def dataset(data='make_classification'):
        if data == 'make_classification':
            X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
        elif data == 'make_moons':
            X, y = make_moons(noise=0.3, random_state=0)

        elif data == 'make_circles':
            X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

        xx, yy = meshgrid(X, y, h=.1)

        return X, y, xx, yy



    def meshgrid(X, y, h=.1):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
        return xx, yy

    X, y, xx, yy = dataset(data='make_classification')

        # Plot the testing points

    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
    #                edgecolors='k')
    params = {'kernel':"linear", 'C': 1000}

    cm = plt.cm.RdBu

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])


    def fit_model(params, X, y, xx, yy):
        clf = SVC(kernel=params['kernel'], C=params['C'], gamma='auto')
        clf.fit(X, y)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        return clf, Z

    # score = clf.score(X_test, y_test)

    def plot(params, X, y, xx, yy):
        contour_axis.clear()
        # just plot the dataset first


        contour_axis.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', cmap=cm_bright, label='Class 1',
                   edgecolors='k')
        contour_axis.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', cmap=cm_bright, label='Class 2',
                   edgecolors='k')

        contour_axis.set_xlabel('x1')
        contour_axis.set_ylabel('x2')
        contour_axis.legend()

        clf, Z = fit_model(params, X, y, xx, yy)


        # Put the result into a color plot

        decision_curve = contour_axis.contourf(xx, yy, Z, cmap=cm, alpha=.5)

        # plot decision boundary and margins
        margins = contour_axis.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

        contour_axis.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=30,
                   linewidth=0.5, marker='x', color='k')



        # plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        #                 size=15, horizontalalignment='right')
    plt.ioff()
    contour_axis = plt.gca()

    # plot(params)


    import ipywidgets as widgets
    from ipywidgets import Button, Layout, HTML, VBox, HBox
    from IPython.display import display, clear_output
    # plt.subplot(121)

    h1 = HTML(value='<b>Datasets</b>')
    w0 = widgets.Dropdown(options=['make_classification', 'make_moons', 'make_circles'],
                            value='make_classification',
                            description='Dataset',
                            disabled = False)



    w1 = widgets.FloatText(description = 'C', value = 10.0)
    w2 = widgets.Dropdown(options=['linear', 'poly', 'rbf', 'sigmoid'],
                            value='linear',
                            description='Kernel',
                            disabled = False)

    h2 = HTML(value='<b>Hyperparameters</b>')

    box_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center')
    button = widgets.Button(description = 'Generate', layout = box_layout)


    grid1 = widgets.VBox(children=(h1, HTML(value=''), w0, HTML(value=''), HTML(value=''), HTML(value=''), h2, HTML(value=''), w1, w2))
    grid1.layout.align_items = 'center'
    grid2 = widgets.VBox(children=(grid1, HTML(value=''), HTML(value=''), HTML(value=''), button))
    grid2.layout.align_items = 'center'
    out=widgets.Output()

    box=widgets.HBox(children=(out, grid2))
    box.layout.align_items = 'center'

    display(box)

    def update_curve(a):
        # New fitting
        params['C'] = w1.value
        params['kernel'] = w2.value
        if w1.value<=0:
            params['C'] = 0.001
            w1.value = 0.001
        X, y, xx, yy = dataset(w0.value)

        clf, Z = fit_model(params, X, y, xx, yy)
        # update curve
        plot(params, X, y, xx, yy)
        with out:
            clear_output(wait=True)
            display(contour_axis.figure)

    # plt.show()
    from IPython.display import display
    # widgets.HBox([kernel_grid, ])
    update_curve(None)

    button.on_click(update_curve)
    
    
    
    
    
def show_knn():

    def dataset(data='make_classification'):
        if data == 'make_classification':
            X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
        elif data == 'make_moons':
            X, y = make_moons(noise=0.3, random_state=0)

        elif data == 'make_circles':
            X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

        xx, yy = meshgrid(X, y, h=.1)

        return X, y, xx, yy



    def meshgrid(X, y, h=.1):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
        return xx, yy

    X, y, xx, yy = dataset(data='make_classification')

        # Plot the testing points

    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
    #                edgecolors='k')
    params = {'metric': "euclidean", 'n_neighbors': 5}

    cm = plt.cm.RdBu

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])


    def fit_model(params, X, y, xx, yy):
        clf = KNeighborsClassifier(metric=params['metric'], n_neighbors=params['n_neighbors'])
        clf.fit(X, y)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        return clf, Z

    # score = clf.score(X_test, y_test)

    def plot(params, X, y, xx, yy):
        contour_axis.clear()
        # just plot the dataset first


        contour_axis.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', cmap=cm_bright, label='Class 1',
                       edgecolors='k')
        contour_axis.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', cmap=cm_bright, label='Class 2',
                       edgecolors='k')

        contour_axis.set_xlabel('x1')
        contour_axis.set_ylabel('x2')
        contour_axis.legend()

        clf, Z = fit_model(params, X, y, xx, yy)


        # Put the result into a color plot

        decision_curve = contour_axis.contourf(xx, yy, Z, cmap=cm, alpha=.3)

        # plot decision boundary and margins
#         margins = contour_axis.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.3,
#                linestyles=['-'])

    #     contour_axis.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=30,
    #                linewidth=0.5, marker='x', color='k')



        # plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        #                 size=15, horizontalalignment='right')
    plt.ioff()
    contour_axis = plt.gca()

    # plot(params)


    import ipywidgets as widgets
    from ipywidgets import Button, Layout, HTML, VBox, HBox
    from IPython.display import display,clear_output
    # plt.subplot(121)

    h1 = HTML(value='<b>Datasets</b>')
    w0 = widgets.Dropdown(options=['make_classification', 'make_moons', 'make_circles'],
                            value='make_classification',
                            description='Dataset',
                            disabled = False)



    w1 = widgets.FloatText(description = 'n_neighbors', value = 5, step=1)
    w2 = widgets.Dropdown(options=['euclidean', 'manhattan', 'chebyshev', 'minkowski','wminkowski' , 'seuclidean', 'mahalanobis'],
                            value='minkowski',
                            description='Metric',
                            disabled = False)

    h2 = HTML(value='<b>Hyperparameters</b>')

    box_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center')
    button = widgets.Button(description = 'Generate', layout = box_layout)


    grid1 = widgets.VBox(children=(h1, HTML(value=''), w0, HTML(value=''), HTML(value=''), HTML(value=''), h2, HTML(value=''), w1, w2))
    grid1.layout.align_items = 'center'
    grid2 = widgets.VBox(children=(grid1, HTML(value=''), HTML(value=''), HTML(value=''), button))
    grid2.layout.align_items = 'center'
    out=widgets.Output()

    box=widgets.HBox(children=(out, grid2))
    box.layout.align_items = 'center'

    display(box)

    def update_curve(a):
        # New fitting
        params['n_neighbors'] = int(w1.value)

        params['metric'] = w2.value
        X, y, xx, yy = dataset(w0.value)
        if params['n_neighbors']<1:
            params['n_neighbors']=1
            w1.value=1
        if params['n_neighbors']>len(X)-1:
            w1.value=len(X)-1
            params['n_neighbors']=len(X)-1
        w1.value=int(w1.value)

        clf, Z = fit_model(params, X, y, xx, yy)
        # update curve
        plot(params, X, y, xx, yy)
        with out:
            clear_output(wait=True)
            display(contour_axis.figure)

    # plt.show()
    from IPython.display import display
    # widgets.HBox([kernel_grid, ])
    update_curve(None)

    button.on_click(update_curve)
    
    
def show_tree():
    # plt.figure(figsize=(10,5))
    def dataset(data='make_classification'):
        if data == 'make_classification':
            X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
        elif data == 'make_moons':
            X, y = make_moons(noise=0.3, random_state=0)

        elif data == 'make_circles':
            X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

        xx, yy = meshgrid(X, y, h=.1)

        return X, y, xx, yy

    params = {'criterion':"gini", 'splitter': 'best', 'max_depth':3}

    cm = plt.cm.RdBu

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])


    def fit_model(params, X, y, xx, yy):
        clf = DecisionTreeClassifier(criterion=params['criterion'], splitter=params['splitter'], max_depth=params['max_depth'])
        clf.fit(X, y)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        return clf, Z

    # score = clf.score(X_test, y_test)

    def plot(params, X, y, xx, yy):
        contour_axis.clear()
        # just plot the dataset first


        contour_axis.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', cmap=cm_bright, label='Class 1',
                       edgecolors='k')
        contour_axis.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', cmap=cm_bright, label='Class 2',
                       edgecolors='k')

        contour_axis.set_xlabel('x0')
        contour_axis.set_ylabel('x1')
        contour_axis.legend()

        clf, Z = fit_model(params, X, y, xx, yy)


        # Put the result into a color plot

        decision_curve = contour_axis.contourf(xx, yy, Z, cmap=cm, alpha=.3)

        # plot decision boundary and margins
    #     margins = contour_axis.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5,
    #            linestyles=['-'])

        plot_tree(clf, ax=ax2)



        # plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        #                 size=15, horizontalalignment='right')
    plt.ioff()
    fig, (contour_axis, ax2)  = plt.subplots(1, 2, figsize=(12,4))
    # contour_axis = plt.gca()

    # fig2, axis2 = plt.subplots(122)
    # plot(params)


    import ipywidgets as widgets
    from ipywidgets import Button, Layout, HTML, VBox, HBox
    from IPython.display import display, clear_output
    # plt.subplot(121)

    h1 = HTML(value='<b>Datasets</b>')
    w0 = widgets.Dropdown(options=['make_classification', 'make_moons', 'make_circles'],
                            value='make_classification',
                            description='Dataset',
                            disabled = False)

    w1 = widgets.Dropdown(options=["gini", "entropy"],
                            value='gini',
                            description='Criterion',
                            disabled = False)
    w2 = widgets.Dropdown(options=["best", "random"],
                            value='best',
                            description='Splitter',
                            disabled = False)
    w3 = widgets.FloatText(description = 'max_depth', value = 3)


    h2 = HTML(value='<b>Hyperparameters</b>')

    box_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center')
    button = widgets.Button(description = 'Generate', layout = box_layout)

    grid0 = widgets.VBox(children=(h1, HTML(value=''), w0, HTML(value=''), HTML(value=''), button))
    grid0.layout.align_items = 'center'

    grid1 = widgets.VBox(children=(h2, HTML(value=''), w1, w2, w3))
    grid1.layout.align_items = 'center'

    # grid2 = widgets.VBox(children=(HTML(value=''), button))
    # grid2.layout.align_items = 'center'

    grid_all = widgets.HBox(children=(grid1, grid0))
    grid_all.layout.align_items = 'center'

    out = widgets.Output()

    box=widgets.VBox(children=(out, grid_all))
    box.layout.align_items = 'center'

    display(box)

    def update_curve(a):
        # New fitting
        params['criterion'] = w1.value
        params['splitter'] = w2.value
        params['max_depth'] = int(w3.value)
        if w3.value<1:
            params['max_depth']=1
            w3.value=1
        w3.value = int(w3.value)
        X, y, xx, yy = dataset(w0.value)

        clf, Z = fit_model(params, X, y, xx, yy)
        # update curve
        plot(params, X, y, xx, yy)
        with out:
            clear_output(wait=True)
            display(contour_axis.figure)
    def meshgrid(X, y, h=.1):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
        return xx, yy

    # plt.show()
    from IPython.display import display
    # widgets.HBox([kernel_grid, ])
    update_curve(None)

    button.on_click(update_curve)
