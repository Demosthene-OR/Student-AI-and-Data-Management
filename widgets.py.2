import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression


# =========================
# Linear Regression Widget
# =========================
def regression_widget():
    np.random.seed(0)
    X = np.linspace(-4, 4, 50)
    y = 0.5*X + 1.5 + np.random.normal(0,1,50)

    # Scatter plot + initial line
    scatter = go.Scatter(x=X, y=y, mode='markers', marker=dict(color='blue'))
    line = go.Scatter(x=X, y=0.5*X + 1.5, mode='lines', line=dict(color='red'))

    fig = go.FigureWidget(data=[scatter, line])
    fig.update_layout(width=400, height=400, title="Linear Regression")

    # Sliders
    beta_slider = widgets.FloatSlider(min=-2, max=2, step=0.1, value=0.5, description="β1")
    bias_slider = widgets.FloatSlider(min=-3, max=3, step=0.1, value=1.5, description="β0")

    def update_plot(beta, bias):
        with fig.batch_update():
            fig.data[1].y = beta*X + bias

    widgets.interact(update_plot, beta=beta_slider, bias=bias_slider)
    display(fig)

# =========================
# Interactive MSE Widget
# =========================
def interactive_MSE():
    np.random.seed(0)
    X = np.linspace(-4, 4, 20)
    y = 1.5*X + np.random.normal(0, 1.2, 20)

    scatter = go.Scatter(x=X, y=y, mode='markers', marker=dict(color='blue'))
    line = go.Scatter(x=X, y=1.5*X, mode='lines', line=dict(color='red'))

    fig = go.FigureWidget(data=[scatter, line])
    fig.update_layout(width=400, height=400, title="Interactive MSE")

    beta_slider = widgets.FloatSlider(min=-2, max=2, step=0.1, value=1.5, description="β1")
    mse_label = widgets.HTML(value=f"<b>MSE: {np.mean((y - 1.5*X)**2):.2f}</b>")

    def update(beta):
        with fig.batch_update():
            y_pred = beta*X
            fig.data[1].y = y_pred
            mse = np.mean((y - y_pred)**2)
            mse_label.value = f"<b>MSE: {mse:.2f}</b>"

    beta_slider.observe(lambda change: update(change['new']), names='value')
    display(widgets.VBox([fig, mse_label, beta_slider]))

# =========================
# Polynomial Regression Widget
# =========================
def polynomial_regression():
    np.random.seed(0)
    n_samples = 50
    X = np.linspace(0, 10, n_samples)
    y = -0.1*X**2 + 1*X + 1.5 + np.random.normal(0,1,n_samples)

    scatter = go.Scatter(x=X, y=y, mode='markers', marker=dict(color='blue'))
    line = go.Scatter(x=X, y=-0.1*X**2 + 1*X + 1.5, mode='lines', line=dict(color='red'))

    fig = go.FigureWidget(data=[scatter, line])
    fig.update_layout(width=400, height=400, title="Polynomial Regression")

    # Sliders
    beta0 = widgets.FloatSlider(-4, 4, 0.1, 0, description="β0")
    beta1 = widgets.FloatSlider(-2, 2, 0.1, 1, description="β1")
    beta2 = widgets.FloatSlider(-2, 2, 0.1, -0.1, description="β2")

    def update(beta0_val, beta1_val, beta2_val):
        with fig.batch_update():
            line.y = beta0_val + beta1_val*X + beta2_val*X**2

    widgets.interact(update, beta0=beta0, beta1=beta1, beta2=beta2)
    display(fig)


def polynomial_regression2():
    np.random.seed(0)
    n_samples = 30
    X = np.linspace(0, 10, n_samples)
    y = -0.1*X**2 + 1*X + 1.5 + np.random.normal(0, 1, n_samples)

    scatter = go.Scatter(x=X, y=y, mode='markers', marker=dict(color='blue'))
    line = go.Scatter(x=X, y=np.zeros_like(X), mode='lines', line=dict(color='red'))

    fig = go.FigureWidget(data=[scatter, line])
    fig.update_layout(width=500, height=400, title="Polynomial Regression Degree d")

    degree_slider = widgets.IntSlider(min=1, max=10, value=2, description="Degree d")

    def update_poly(d):
        # Create polynomial features
        data = X.reshape(-1,1)
        for i in range(2, d+1):
            data = np.hstack([data, X.reshape(-1,1)**i])
        # Fit linear regression
        lr = LinearRegression()
        lr.fit(data, y)
        y_pred = lr.predict(data)
        with fig.batch_update():
            fig.data[1].y = y_pred

    widgets.interact(update_poly, d=degree_slider)
    display(fig)


