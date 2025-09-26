import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from IPython.display import display

# Détection Colab
def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# === Version Matplotlib (fallback pour Colab) ===
def regression_widget():
    # Données simulées
    np.random.seed(0)
    n_samples = 50
    X = np.linspace(-4, 4, n_samples).reshape(-1, 1)
    y = 0.5 * X[:, 0] + 1.5 + np.random.normal(0, 1, n_samples)

    # Modèle initial
    fig, ax = plt.subplots()
    sc = ax.scatter(X, y, color="blue", alpha=0.7)
    line, = ax.plot(X, X, color="red", lw=2)  # droite initiale

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


