import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML
from sklearn.linear_model import LinearRegression


# Détection Colab
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


# =========================================================
# 1. Simple Linear Regression Widget
# =========================================================
def regression_widget():
    n_samples = 100
    alpha, bias = 0.5, 1.5
    X = np.linspace(-4, 4, n_samples)
    y = alpha * X + bias + np.random.normal(0, 1, n_samples)

    try:
        import bqplot.pyplot as plt

        x_sc = plt.LinearScale(min=-4, max=4)
        y_sc = plt.LinearScale(min=-4, max=8)

        scatter = plt.Scatter(x=X, y=y, colors=['blue'], scales={'x': x_sc, 'y': y_sc})
        line = plt.Lines(x=[-4, 4], y=[-4, 4], colors=['red'], scales={'x': x_sc, 'y': y_sc})

        fig = plt.Figure(marks=[scatter, line], axes=[plt.Axis(scale=x_sc), plt.Axis(scale=y_sc)])
        fig.layout.height, fig.layout.width = '400px', '400px'
        display(fig)

        @widgets.interact(beta=widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0),
                          bias=widgets.FloatSlider(min=-4, max=4, step=0.1, value=0.0))
        def update(beta, bias):
            line.y = [beta * -4 + bias, beta * 4 + bias]

    except Exception:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(X, y, c='blue', s=10)
        (line,) = ax.plot([-4, 4], [-4, 4], 'r-')

        def update(beta=1.0, bias=0.0):
            line.set_ydata([beta * -4 + bias, beta * 4 + bias])
            fig.canvas.draw_idle()

        sliders = widgets.interactive(update,
                                      beta=(-4, 4, 0.1),
                                      bias=(-4, 4, 0.1))
        display(widgets.VBox([sliders, fig.canvas]))


# =========================================================
# 2. Interactive MSE
# =========================================================
def interactive_MSE():
    np.random.seed(3)
    X = np.linspace(-4, 4, 20)
    y = 0.7 * X + np.random.normal(0, 1.2, len(X))

    try:
        import bqplot.pyplot as plt

        x_sc, y_sc = plt.LinearScale(min=-4, max=4), plt.LinearScale(min=-6, max=6)
        scatter = plt.Scatter(x=X, y=y, colors=['blue'], scales={'x': x_sc, 'y': y_sc})
        line = plt.Lines(x=[-4, 4], y=[-4, 4], colors=['red'], scales={'x': x_sc, 'y': y_sc})
        fig = plt.Figure(marks=[scatter, line], axes=[plt.Axis(scale=x_sc), plt.Axis(scale=y_sc)])
        fig.layout.height, fig.layout.width = '400px', '400px'
        display(fig)

        @widgets.interact(beta=widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.0))
        def update(beta):
            line.y = [beta * -4, beta * 4]

    except Exception:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(X, y, c='blue', s=10)
        (line,) = ax.plot([-4, 4], [-4, 4], 'r-')

        def update(beta=1.0):
            line.set_ydata([beta * -4, beta * 4])
            fig.canvas.draw_idle()

        sliders = widgets.interactive(update, beta=(-4, 4, 0.1))
        display(widgets.VBox([sliders, fig.canvas]))


# =========================================================
# 3. Quadratic Regression
# =========================================================
def polynomial_regression():
    n_samples = 100
    X = np.linspace(0, 10, n_samples)
    y = -0.1 * X**2 + 1 * X + 1.5 + np.random.normal(0, 1, n_samples)

    try:
        import bqplot.pyplot as plt

        x_sc, y_sc = plt.LinearScale(min=0, max=10), plt.LinearScale(min=-5, max=15)
        scatter = plt.Scatter(x=X, y=y, colors=['blue'], scales={'x': x_sc, 'y': y_sc})
        line = plt.Lines(x=X, y=X, colors=['red'], scales={'x': x_sc, 'y': y_sc})
        fig = plt.Figure(marks=[scatter, line], axes=[plt.Axis(scale=x_sc), plt.Axis(scale=y_sc)])
        fig.layout.height, fig.layout.width = '400px', '400px'
        display(fig)

        @widgets.interact(b0=(-4, 4, 0.1), b1=(-2, 2, 0.1), b2=(-2, 2, 0.1))
        def update(b0=0, b1=1, b2=0):
            line.y = b0 + b1 * X + b2 * X**2

    except Exception:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(X, y, c='blue', s=10)
        (line,) = ax.plot(X, X, 'r-')

        def update(b0=0, b1=1, b2=0):
            line.set_ydata(b0 + b1 * X + b2 * X**2)
            fig.canvas.draw_idle()

        sliders = widgets.interactive(update, b0=(-4, 4, 0.1), b1=(-2, 2, 0.1), b2=(-2, 2, 0.1))
        display(widgets.VBox([sliders, fig.canvas]))


# =========================================================
# 4. Polynomial Regression (variable degree)
# =========================================================
def polynomial_regression2():
    n_samples = 30
    X = np.linspace(0, 10, n_samples)
    y = -0.1 * X**2 + 1 * X + 1.5 + np.random.normal(0, 1, n_samples)

    try:
        import bqplot.pyplot as plt

        x_sc, y_sc = plt.LinearScale(min=0, max=10), plt.LinearScale(min=-5, max=15)
        scatter = plt.Scatter(x=X, y=y, colors=['blue'], scales={'x': x_sc, 'y': y_sc})
        line = plt.Lines(x=X, y=y, colors=['red'], scales={'x': x_sc, 'y': y_sc})
        fig = plt.Figure(marks=[scatter, line], axes=[plt.Axis(scale=x_sc), plt.Axis(scale=y_sc)])
        fig.layout.height, fig.layout.width = '400px', '400px'
        display(fig)

        @widgets.interact(d=(1, 10, 1))
        def update(d=2):
            data = X.reshape(-1, 1)
            for i in range(2, d + 1):
                data = np.hstack([data, (X**i).reshape(-1, 1)])
            lr = LinearRegression().fit(data, y)
            line.y = lr.predict(data)

    except Exception:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(X, y, c='blue', s=10)
        (line,) = ax.plot(X, y, 'r-')

        def update(d=2):
            data = X.reshape(-1, 1)
            for i in range(2, d + 1):
                data = np.hstack([data, (X**i).reshape(-1, 1)])
            lr = LinearRegression().fit(data, y)
            line.set_ydata(lr.predict(data))
            fig.canvas.draw_idle()

        sliders = widgets.interactive(update, d=(1, 10, 1))
        display(widgets.VBox([sliders, fig.canvas]))
