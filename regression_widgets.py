import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML
import bqplot.pyplot as bplt

# Détection Colab
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

# ===============================
# 1. Régression linéaire simple
# ===============================
def regression_widget_linear():
    X = np.linspace(-4, 4, 100)
    y = 0.5*X + 1.5 + np.random.normal(0,1,len(X))

    # Scales
    x_sc = bplt.LinearScale(min=-4, max=4)
    y_sc = bplt.LinearScale(min=min(y)-1, max=max(y)+1)

    # Scatter et ligne
    scatter = bplt.Scatter(x=X, y=y, colors=['blue'], scales={'x': x_sc, 'y': y_sc})
    line = bplt.Lines(x=X, y=0.5*X+1.5, colors=['red'], scales={'x': x_sc, 'y': y_sc})

    # Axes avec grille
    x_ax = bplt.Axis(scale=x_sc, label='X', grid_lines='solid')
    y_ax = bplt.Axis(scale=y_sc, orientation='vertical', label='Y', grid_lines='solid')

    # Figure
    fig = bplt.Figure(marks=[scatter, line], axes=[x_ax, y_ax])
    fig.layout.height = '400px'
    fig.layout.width = '400px'

    # Sliders
    beta0 = widgets.FloatSlider(min=-4, max=4, step=0.1, value=1.5, description="β0")
    beta1 = widgets.FloatSlider(min=-4, max=4, step=0.1, value=0.5, description="β1")

    # Fonction de mise à jour
    def update(change):
        line.y = beta1.value*X + beta0.value
        # Ne pas toucher y_sc.min/max ici
        # Bqplot ajuste automatiquement l'affichage

    # Connecter sliders
    beta0.observe(update, names='value')
    beta1.observe(update, names='value')

    display(fig, widgets.VBox([beta0, beta1]))


# ===============================
# 2. MSE interactif
# ===============================
def interactive_MSE_widget():
    np.random.seed(3)
    X = np.linspace(-4, 4, 20)
    y = 0.7*X + np.random.normal(0,1.2,len(X))

    x_sc = bplt.LinearScale(min=-4, max=4)
    y_sc = bplt.LinearScale(min=min(y)-1, max=max(y)+1)

    scatter = bplt.Scatter(x=X, y=y, colors=['blue'], scales={'x': x_sc, 'y': y_sc})
    line = bplt.Lines(x=X, y=0.7*X, colors=['red'], scales={'x': x_sc, 'y': y_sc})

    fig = bplt.Figure(marks=[scatter, line], axes=[bplt.Axis(scale=x_sc), bplt.Axis(scale=y_sc)])
    fig.layout.height = '400px'
    fig.layout.width = '400px'

    beta1 = widgets.FloatSlider(min=-4, max=4, step=0.1, value=0.7, description="β1")

    def update(change):
        line.y = beta1.value * X

    beta1.observe(update, names='value')

    display(fig, beta1)

# ===============================
# 3. Régression quadratique
# ===============================
def polynomial_regression_widget():
    X = np.linspace(0, 10, 50)
    y = -0.1*X**2 + X + 1.5 + np.random.normal(0,1,len(X))

    x_sc = bplt.LinearScale(min=0, max=10)
    y_sc = bplt.LinearScale(min=min(y)-1, max=max(y)+1)

    scatter = bplt.Scatter(x=X, y=y, colors=['blue'], scales={'x': x_sc, 'y': y_sc})
    line = bplt.Lines(x=X, y=-0.1*X**2 + X + 1.5, colors=['red'], scales={'x': x_sc, 'y': y_sc})

    fig = bplt.Figure(marks=[scatter, line], axes=[bplt.Axis(scale=x_sc), bplt.Axis(scale=y_sc)])
    fig.layout.height = '400px'
    fig.layout.width = '400px'

    b0 = widgets.FloatSlider(min=-4, max=4, step=0.1, value=0.0, description="β0")
    b1 = widgets.FloatSlider(min=-2, max=2, step=0.1, value=1.0, description="β1")
    b2 = widgets.FloatSlider(min=-2, max=2, step=0.1, value=-0.1, description="β2")

    def update(change):
        line.y = b0.value + b1.value*X + b2.value*X**2

    for w in [b0, b1, b2]:
        w.observe(update, names='value')

    display(fig, widgets.VBox([b0, b1, b2]))

# ===============================
# 4. Régression polynomiale variable
# ===============================
from sklearn.linear_model import LinearRegression

def polynomial_regression_variable_widget():
    X = np.linspace(0, 10, 30)
    y = -0.1*X**2 + X + 1.5 + np.random.normal(0,1,len(X))

    x_sc = bplt.LinearScale(min=0, max=10)
    y_sc = bplt.LinearScale(min=min(y)-1, max=max(y)+1)

    scatter = bplt.Scatter(x=X, y=y, colors=['blue'], scales={'x': x_sc, 'y': y_sc})
    line = bplt.Lines(x=X, y=-0.1*X**2 + X + 1.5, colors=['red'], scales={'x': x_sc, 'y': y_sc})

    fig = bplt.Figure(marks=[scatter, line], axes=[bplt.Axis(scale=x_sc), bplt.Axis(scale=y_sc)])
    fig.layout.height = '400px'
    fig.layout.width = '400px'

    degree = widgets.IntSlider(min=1, max=10, value=2, description="Degree d")

    def update(change):
        data = X.reshape(-1,1)
        for i in range(2, degree.value+1):
            data = np.hstack([data, (X**i).reshape(-1,1)])
        lr = LinearRegression().fit(data, y)
        line.y = lr.predict(data)

    degree.observe(update, names='value')

    display(fig, degree)
