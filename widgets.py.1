import numpy as np
import ipywidgets as widgets
from IPython.display import display
from sklearn.linear_model import LinearRegression

# =========================
# Détection Colab
# =========================
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# =========================
# Linear Regression Widget
# =========================
def regression_widget():
    np.random.seed(0)
    X = np.linspace(-4, 4, 50).reshape(-1, 1)
    y = 0.5 * X[:, 0] + 1.5 + np.random.normal(0, 1, 50)

    if in_colab():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))  # équivalent 400px x 400px
        sc = ax.scatter(X, y, color="blue", alpha=0.7)
        line, = ax.plot(X, X, color="red", lw=2)
        ax.set_xlim(-5, 5)
        ax.set_ylim(min(y) - 2, max(y) + 2)
        ax.set_title("Linear Regression (interactive)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        beta_slider = widgets.FloatSlider(min=-2, max=2, step=0.1, value=0.5, description="β1")
        bias_slider = widgets.FloatSlider(min=-3, max=3, step=0.1, value=1.5, description="β0")

        def update_plot(beta, bias):
            line.set_ydata(beta * X[:, 0] + bias)
            fig.canvas.draw_idle()

        widgets.interact(update_plot, beta=beta_slider, bias=bias_slider)
        display(fig)

    else:
        import bqplot.pyplot as plt_bq
        x_sc, y_sc = plt_bq.LinearScale(), plt_bq.LinearScale()
        scatter = plt_bq.Scatter(x=X[:, 0], y=y, scales={'x': x_sc, 'y': y_sc})
        line = plt_bq.Lines(x=X[:, 0], y=0.5*X[:, 0] + 1.5, scales={'x': x_sc, 'y': y_sc}, colors=["red"])
        fig = plt_bq.Figure(marks=[scatter, line],
                            axes=[plt_bq.Axis(scale=x_sc), plt_bq.Axis(scale=y_sc, orientation="vertical")])
        fig.layout.height = '400px'
        fig.layout.width = '400px'

        @widgets.interact(beta=(-2, 2, 0.1), bias=(-3, 3, 0.1))
        def update(beta=0.5, bias=1.5):
            line.y = beta * X[:, 0] + bias

        display(fig)

# =========================
# Interactive MSE Widget
# =========================
def interactive_MSE():
    np.random.seed(0)
    X = np.linspace(-4, 4, 20).reshape(-1, 1)
    y = 1.5 * X[:, 0] + np.random.normal(0, 1.2, 20)

    if in_colab():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        sc = ax.scatter(X, y, color="blue", alpha=0.7)
        line, = ax.plot(X, X, color="red", lw=2)
        ax.set_xlim(-5, 5)
        ax.set_ylim(min(y) - 2, max(y) + 2)
        ax.set_title("Interactive MSE")

        beta_slider = widgets.FloatSlider(min=-2, max=2, step=0.1, value=1.5, description="β1")

        def update(beta):
            y_pred = beta * X[:, 0]
            line.set_ydata(y_pred)
            mse = np.mean((y - y_pred) ** 2)
            ax.set_title(f"MSE = {mse:.2f}")
            fig.canvas.draw_idle()

        widgets.interact(update, beta=beta_slider)
        display(fig)

    else:
        import bqplot.pyplot as plt_bq
        x_sc, y_sc = plt_bq.LinearScale(), plt_bq.LinearScale()
        scatter = plt_bq.Scatter(x=X[:, 0], y=y, scales={'x': x_sc, 'y': y_sc})
        line = plt_bq.Lines(x=X[:, 0], y=1.5*X[:, 0], scales={'x': x_sc, 'y': y_sc}, colors=["red"])
        fig = plt_bq.Figure(marks=[scatter, line],
                            axes=[plt_bq.Axis(scale=x_sc), plt_bq.Axis(scale=y_sc, orientation="vertical")])
        fig.layout.height = '400px'
        fig.layout.width = '400px'

        @widgets.interact(beta=(-2, 2, 0.1))
        def update(beta=1.5):
            line.y = beta * X[:, 0]

        display(fig)

# =========================
# Polynomial Regression Widget
# =========================
def polynomial_regression():
    np.random.seed(0)
    n_samples = 50
    X = np.linspace(0, 10, n_samples)
    y = -0.1*X**2 + 1*X + 1.5 + np.random.normal(0,1,n_samples)

    if in_colab():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        sc = ax.scatter(X, y, color="blue", alpha=0.7)
        line, = ax.plot(X, X, color="red", lw=2)
        ax.set_xlim(0, 10)
        ax.set_ylim(min(y)-2, max(y)+2)
        ax.set_title("Polynomial Regression (interactive)")

        b0 = widgets.FloatSlider(-4, 4, 0.1, 0, description="β0")
        b1 = widgets.FloatSlider(-2, 2, 0.1, 1, description="β1")
        b2 = widgets.FloatSlider(-2, 2, 0.1, -0.1, description="β2")

        def update_plot(beta0, beta1, beta2):
            y_pred = beta0 + beta1*X + beta2*X**2
            line.set_ydata(y_pred)
            fig.canvas.draw_idle()

        widgets.interact(update_plot, beta0=b0, beta1=b1, beta2=b2)
        display(fig)

    else:
        import bqplot.pyplot as plt_bq
        x_sc, y_sc = plt_bq.LinearScale(), plt_bq.LinearScale()
        scatter = plt_bq.Scatter(x=X, y=y, scales={'x': x_sc, 'y': y_sc})
        line = plt_bq.Lines(x=X, y=-0.1*X**2 + 1*X + 1.5, scales={'x': x_sc, 'y': y_sc}, colors=["red"])
        fig = plt_bq.Figure(marks=[scatter, line],
                            axes=[plt_bq.Axis(scale=x_sc), plt_bq.Axis(scale=y_sc, orientation="vertical")])
        fig.layout.height = '400px'
        fig.layout.width = '400px'

        @widgets.interact(beta0=(-4,4,0.1), beta1=(-2,2,0.1), beta2=(-2,2,0.1))
        def update(beta0=0, beta1=1, beta2=-0.1):
            line.y = beta0 + beta1*X + beta2*X**2

        display(fig)

