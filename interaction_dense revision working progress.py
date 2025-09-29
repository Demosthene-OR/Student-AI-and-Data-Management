# file: portable_visuals.py
# Compatible Colab & Jupyter : Plotly + ipywidgets (fallback matplotlib when useful)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML
from sklearn import datasets
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, mean_squared_error, hinge_loss
from sklearn.utils import shuffle
from sklearn.datasets import make_moons
from scipy.interpolate import interp1d
from scipy import ndimage
import json, os
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt

# Helper: detect Colab
def in_colab():
    try:
        import google.colab
        return True
    except Exception:
        return False

def enable_colab_widgets():
    if in_colab():
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
        except Exception:
            display(HTML("<b style='color:red'>⚠️ Attention :</b> Widgets may not be fully supported in this Colab instance."))

enable_colab_widgets()

# ---------------------------
# Utility helpers for Plotly
# ---------------------------
def figwidget_2d_scatter_line(x_scatter, y_scatter, x_line, y_line, title="Figure", x_title="x", y_title="y", width=700, height=450):
    """Return a Plotly FigureWidget with scatter and line traces (for interactive updates)."""
    fw = go.FigureWidget()
    fw.add_scatter(x=x_scatter, y=y_scatter, mode='markers', marker=dict(size=6, color='blue'), name='data')
    fw.add_scatter(x=x_line, y=y_line, mode='lines', line=dict(color='red'), name='model')
    fw.update_layout(title=title, width=width, height=height, xaxis_title=x_title, yaxis_title=y_title, margin=dict(l=20,r=20,b=40,t=40))
    return fw

def surface_figure_from_plane(xx, yy, zz, scatter3d=None, title="3D plane", width=700, height=500):
    fw = go.FigureWidget()
    fw.add_surface(x=xx, y=yy, z=zz, opacity=0.5, showscale=False)
    if scatter3d is not None:
        fw.add_trace(go.Scatter3d(x=scatter3d['x'], y=scatter3d['y'], z=scatter3d['z'], mode='markers',
                                 marker=dict(size=scatter3d.get('size',4), color=scatter3d.get('color',scatter3d.get('c',None)), colorscale='Viridis')))
    fw.update_layout(title=title, width=width, height=height, margin=dict(l=0,r=0,b=0,t=30))
    return fw

# ---------------------------
# show_dotProduct (converted to Plotly)
# ---------------------------

def show_dotProduct():
    # Valeurs initiales
    x1_0, x2_0, w1_0, w2_0 = -2.0, 1.0, 0.0, 2.0

    # Sliders
    s_x1 = widgets.FloatSlider(min=-4,max=4,step=0.1,value=x1_0, description='x1')
    s_x2 = widgets.FloatSlider(min=-4,max=4,step=0.1,value=x2_0, description='x2')
    s_w1 = widgets.FloatSlider(min=-4,max=4,step=0.1,value=w1_0, description='w1')
    s_w2 = widgets.FloatSlider(min=-4,max=4,step=0.1,value=w2_0, description='w2')

    def update(x1, x2, w1, w2):
        x = np.array([x1, x2])
        w = np.array([w1, w2])
        normw = np.linalg.norm(w) if np.linalg.norm(w)>0 else 1.0
        proj = (np.dot(w,x)/normw**2) * w
        dot = np.dot(w,x)

        # recréer la figure à chaque update
        fig = go.Figure(
            layout=go.Layout(
                title="Dot product / projection",
                width=700, height=500,
                xaxis=dict(range=[-5, 5], zeroline=True, showgrid=True),
                yaxis=dict(range=[-5, 5], zeroline=True, showgrid=True),
                showlegend=False
            )
        )
        fig.add_trace(go.Scatter(x=[x1], y=[x2], mode='markers+text',
                                  text=["x"], textposition="bottom right",
                                  marker=dict(size=10, color='blue')))
        fig.add_trace(go.Scatter(x=[0, w1], y=[0, w2], mode='lines+markers+text',
                                  text=["","w"], textposition="top right",
                                  line=dict(color='green'), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=[proj[0], x1], y=[proj[1], x2],
                                  mode='lines', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=[proj[0]], y=[proj[1]], mode='markers+text',
                                  text=[f"⟨w,x⟩={np.round(dot,3)}"],
                                  textposition="top left",
                                  marker=dict(size=8, color='red')))
        fig.show()

    out = widgets.interactive_output(update, {'x1': s_x1, 'x2': s_x2, 'w1': s_w1, 'w2': s_w2})
    display(widgets.HBox([widgets.VBox([s_x1, s_x2, s_w1, s_w2]), widgets.VBox([out])]))





# ---------------------------
# show_data (2D scatter of Iris)
# ---------------------------
def show_data():
    iris_X = datasets.load_iris()['data'][:,:2]
    iris_y = datasets.load_iris()['target']
    iris_y[iris_y==2]=1
    colors = ['green' if yi==0 else 'orange' for yi in iris_y]
    scaled = (iris_X - np.mean(iris_X, axis=0)) / np.std(iris_X, axis=0)
    fig = go.FigureWidget()
    fig.add_scatter(x=scaled[:,0], y=scaled[:,1]-1, mode='markers', marker=dict(color=colors, size=6))
    fig.update_layout(title="Iris (scaled 2D)", width=700, height=500, xaxis=dict(title='Sepal Length'), yaxis=dict(title='Sepal Width'))
    display(fig)

# ---------------------------
# show_losses (loss vs rotation angle)  — converted to Plotly
# ---------------------------
def show_losses():
    iris_X = datasets.load_iris()['data'][:,:2]
    iris_y = datasets.load_iris()['target']
    iris_y[iris_y==2]=1
    iris_y = 1 - iris_y  # follow original
    scaled = (iris_X - np.mean(iris_X, axis=0)) / np.std(iris_X, axis=0)
    lr = LogisticRegression(fit_intercept=False).fit(scaled, iris_y)

    def change_angle_and_predict(alpha_deg):
        a = alpha_deg*np.pi/180
        x = -np.sin(a); y = np.cos(a)
        lr.coef_[0,:] = [x,y]
        return lr

    betas = np.linspace(-90, 120, 500)
    data_loss = [log_loss(iris_y, change_angle_and_predict(b).predict_proba(scaled)) for b in betas]
    data_mse = [mean_squared_error(iris_y, change_angle_and_predict(b).predict_proba(scaled)[:,1]) for b in betas]
    data_hinge = [hinge_loss(iris_y*2-1, change_angle_and_predict(b).decision_function(scaled)) for b in betas]

    # Build a subplot: left = scatter (iris), right-top = loss, right-bottom = mse/hinge
    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan":2}, {"type":"xy"}],[None, {"type":"xy"}]],
                        subplot_titles=("Iris data", "Loss (cross-entropy)", "MSE / Hinge"))
    # Iris data (left)
    colors = ['green' if yi==0 else 'orange' for yi in iris_y]
    fig.add_trace(go.Scatter(x=scaled[:,0], y=scaled[:,1]-1, mode='markers', marker=dict(color=colors, size=6)), row=1, col=1)

    # Loss trace
    loss_trace = go.Scatter(x=betas, y=data_loss, mode='lines', name='loss', line=dict(color='green'))
    loss_dot = go.Scatter(x=[0], y=[data_loss[0]], mode='markers', marker=dict(color='red', size=10), name='loss_dot')
    fig.add_trace(loss_trace, row=1, col=2)
    fig.add_trace(loss_dot, row=1, col=2)

    # MSE trace
    mse_trace = go.Scatter(x=betas, y=data_mse, mode='lines', name='mse', line=dict(color='blue'))
    mse_dot = go.Scatter(x=[0], y=[data_mse[0]], mode='markers', marker=dict(color='red', size=10), name='mse_dot')
    fig.add_trace(mse_trace, row=2, col=2)
    fig.add_trace(mse_dot, row=2, col=2)

    fig.update_layout(height=600, width=1000, title_text="Losses and metrics vs rotation angle (deg)")
    widget = go.FigureWidget(fig)
    display(widget)

    # slider
    slider = widgets.IntSlider(min=-90, max=120, step=1, value=0, description='β (deg)')
    def update_slider(beta):
        # recompute dots quickly using precomputed data arrays
        idx = int(np.searchsorted(betas, beta))
        if idx < 0: idx=0
        if idx >= len(betas): idx=len(betas)-1
        with widget.batch_update():
            widget.data[1].x = [betas[idx]]; widget.data[1].y = [data_loss[idx]]  # loss_dot
            widget.data[3].x = [betas[idx]]; widget.data[3].y = [data_mse[idx]]   # mse_dot
    out = widgets.interactive_output(update_slider, {'beta': slider})
    display(widgets.HBox([slider, out]))

# ---------------------------
# show_accuracy (similar to show_losses but showing accuracy)
# ---------------------------
def show_accuracy():
    iris_X = datasets.load_iris()['data'][:,:2]
    iris_y = datasets.load_iris()['target']
    iris_y[iris_y==2] = 1
    iris_y = 1 - iris_y
    scaled = (iris_X - np.mean(iris_X, axis=0)) / np.std(iris_X, axis=0)
    lr = LogisticRegression(fit_intercept=False).fit(scaled, iris_y)
    betas = np.linspace(-90, 120, 500)

    def acc_for_beta(b):
        a = b*np.pi/180
        x,y = -np.sin(a), np.cos(a)
        lr.coef_[0,:] = [x,y]
        return lr.score(scaled, iris_y)

    data_acc = [acc_for_beta(b) for b in betas]
    data_loss = [log_loss(iris_y, change_pred := LogisticRegression(fit_intercept=False).fit(scaled, iris_y).predict_proba(scaled)) for _ in betas]  # placeholder
    # Use simple figure for accuracy
    fig = go.FigureWidget()
    fig.add_scatter(x=betas, y=data_acc, mode='lines', name='accuracy', line=dict(color='green'))
    dot = go.Scatter(x=[0], y=[data_acc[0]], mode='markers', marker=dict(color='red', size=10))
    fig.add_trace(dot)
    fig.update_layout(title="Accuracy vs β (rotation)", width=800, height=400, xaxis_title='β (deg)')
    display(fig)

    slider = widgets.IntSlider(min=-90, max=120, step=1, value=0, description='β')
    def update(beta):
        idx = int(np.searchsorted(betas, beta))
        if idx < 0: idx=0
        if idx>=len(betas): idx=len(betas)-1
        with fig.batch_update():
            fig.data[1].x = [betas[idx]]; fig.data[1].y = [data_acc[idx]]
    out = widgets.interactive_output(update, {'beta': slider})
    display(widgets.HBox([slider, out]))

# ---------------------------
# show_plane (converted to Plotly 3D)
# ---------------------------
def show_plane(n=50, w=None, b=0.0):
    """
    3D plane visualization with points colored by sign of w·x + b.
    Uses Plotly FigureWidget for interactivity.
    """
    np.random.seed(42)
    if w is None:
        w = np.array([1.0, 1.0, 1.0])
    else:
        w = np.asarray(w)
        if w.size == 2:
            w = np.array([w[0], w[1], 1.0])

    # random cloud
    scat = np.random.normal(0,5,(n,3))
    cross_prods = scat @ w + b
    colors = ['blue' if cp>=0 else 'green' for cp in cross_prods]

    # Build a parametric plane mesh centered around origin using two basis vectors orthogonal to w
    # find two orthonormal basis vectors u,v in plane orthogonal to w
    w_norm = w / np.linalg.norm(w)
    # choose arbitrary vector not collinear
    a = np.array([1,0,0])
    if np.allclose(np.abs(np.dot(a,w_norm)),1):
        a = np.array([0,1,0])
    u = np.cross(w_norm, a)
    u = u / np.linalg.norm(u)
    v = np.cross(w_norm, u)
    # grid
    lin = np.linspace(-10,10,30)
    U, V = np.meshgrid(lin, lin)
    XX = (np.outer(U.ravel(), u) + np.outer(V.ravel(), v)).reshape(U.shape + (3,))
    # plane points coordinates
    XXx = XX[:,:,0]; XXy = XX[:,:,1]; XXz = XX[:,:,2] - b/w[2] if w[2]!=0 else XX[:,:,2]

    # Plotly figure
    fig = go.FigureWidget()
    fig.add_trace(go.Surface(x=XXx, y=XXy, z=XXz, opacity=0.5, showscale=False, colorscale='RdBu'))
    fig.add_trace(go.Scatter3d(x=scat[:,0], y=scat[:,1], z=scat[:,2], mode='markers',
                               marker=dict(size=4, color=['red' if c<0 else 'blue' for c in cross_prods])))
    fig.update_layout(title="3D plane & point cloud", width=900, height=600, scene=dict(xaxis=dict(title='x'), yaxis=dict(title='y'), zaxis=dict(title='z')))
    display(fig)

    # interactive controls: sliders for w1,w2,w3,bias
    s_w1 = widgets.FloatSlider(min=-5,max=5,step=0.1,value=float(w[0]), description='w1')
    s_w2 = widgets.FloatSlider(min=-5,max=5,step=0.1,value=float(w[1]), description='w2')
    s_w3 = widgets.FloatSlider(min=-5,max=5,step=0.1,value=float(w[2]), description='w3')
    s_b = widgets.FloatSlider(min=-20,max=20,step=0.1,value=float(b), description='bias')

    def update(w1,w2,w3,bias):
        W = np.array([w1,w2,w3])
        # recompute basis and plane
        w_norm = W/ (np.linalg.norm(W) if np.linalg.norm(W)>0 else 1.0)
        a = np.array([1,0,0])
        if np.allclose(np.abs(np.dot(a,w_norm)),1):
            a = np.array([0,1,0])
        u = np.cross(w_norm, a); u = u/np.linalg.norm(u)
        v = np.cross(w_norm, u)
        lin = np.linspace(-10,10,30)
        U,V = np.meshgrid(lin, lin)
        XX = (np.outer(U.ravel(), u) + np.outer(V.ravel(), v)).reshape(U.shape + (3,))
        XXx = XX[:,:,0]; XXy = XX[:,:,1]; 
        if W[2]==0:
            XXz = XX[:,:,2]
        else:
            XXz = XX[:,:,2] - bias/W[2]
        # update surface & scatter colors (recompute cross_prods)
        scat_local = scat.copy()
        cross_prods_local = scat_local @ W + bias
        colors_local = ['red' if cp<0 else 'blue' for cp in cross_prods_local]
        with fig.batch_update():
            fig.data[0].x = XXx; fig.data[0].y = XXy; fig.data[0].z = XXz
            fig.data[1].x = scat_local[:,0]; fig.data[1].y = scat_local[:,1]; fig.data[1].z = scat_local[:,2]
            fig.data[1].marker.color = colors_local

    out = widgets.interactive_output(update, {'w1': s_w1, 'w2': s_w2, 'w3': s_w3, 'bias': s_b})
    display(widgets.HBox([widgets.VBox([s_w1, s_w2, s_w3, s_b]), out]))

# ---------------------------
# show_optimization (non convex example)
# ---------------------------
def show_optimization():
    def f2(xs): 
        x = xs/5.0
        return x**4 + x**3 - 6*x**2 + 1
    def f2_p(xs):
        x = xs/5.0
        return 4*(x**3) + 3*(x**2) - 12*x

    xs = np.linspace(-40, 40, 400)
    ys = f2(xs)

    fig = go.FigureWidget()
    fig.add_scatter(x=xs, y=ys, mode='lines', line=dict(color='red'))
    pt = go.Scatter(x=[-5], y=[f2(-5)], mode='markers', marker=dict(size=8, color='green'))
    fig.add_trace(pt)
    fig.update_layout(title="Non-convex function and gradient descent", width=900, height=500)
    display(fig)

    # controls
    s_x0 = widgets.FloatSlider(min=-15, max=11, step=0.5, value=-5, description='x0')
    s_lr = widgets.BoundedFloatText(min=0.001, max=0.6, step=0.01, value=0.1, description='lr')
    play = widgets.Play(value=0, min=0, max=100, step=1, interval=100)
    step = widgets.IntSlider(min=0, max=100, value=0, description='step')
    widgets.jslink((play,'value'), (step,'value'))

    def update(x0, lr, stepv):
        # build trajectory
        xk = np.zeros(101)
        xk[0] = x0
        for k in range(1,101):
            xk[k] = xk[k-1] - lr * f2_p(xk[k-1])
        with fig.batch_update():
            fig.data[1].x = xk[:stepv+1]
            fig.data[1].y = f2(xk[:stepv+1])

    out = widgets.interactive_output(update, {'x0': s_x0, 'lr': s_lr, 'stepv': step})
    display(widgets.VBox([widgets.HBox([s_x0, s_lr]), widgets.HBox([play, step]), out]))

# ---------------------------
# show_optimization_square / show_gradient_descent (convex)
# ---------------------------
def show_optimization_square():
    xs = np.linspace(-10, 10, 400)
    ys = xs**2 + 2
    fig = go.FigureWidget()
    fig.add_scatter(x=xs, y=ys, mode='lines')
    pt = go.Scatter(x=[-5], y=[(-5)**2 + 2], mode='markers', marker=dict(color='green', size=8))
    fig.add_trace(pt)
    fig.update_layout(title="Quadratic optimization", width=800, height=450)
    display(fig)
    slider = widgets.FloatSlider(min=-10,max=10,step=0.2,value=-5, description='x')
    def update(x):
        fig.data[1].x = [x]; fig.data[1].y = [x**2 +2]
    out = widgets.interactive_output(update, {'x': slider})
    display(widgets.HBox([slider, out]))

# ---------------------------
# show_dataset_moon
# ---------------------------
def show_dataset_moon():
    X, y = make_moons(n_samples=100, noise=0.12)
    fig = go.FigureWidget()
    fig.add_scatter(x=X[y==0,0], y=X[y==0,1], mode='markers', marker=dict(color='blue'), name='class 0')
    fig.add_scatter(x=X[y==1,0], y=X[y==1,1], mode='markers', marker=dict(color='red'), name='class 1')
    fig.update_layout(title="Moons dataset", width=700, height=500)
    display(fig)

# ---------------------------
# show_loss (3D loss landscape) - plotly
# ---------------------------
def show_loss(resolution=50, x_range=[-10,10], y_range=[-10,10]):
    iris_X = datasets.load_iris()['data'][:,:2]
    iris_y = datasets.load_iris()['target']; iris_y[iris_y==2]=1
    scaled = (iris_X - iris_X.mean(axis=0)) / iris_X.std(axis=0)
    def loss_fn(w1,w2):
        preds = (w1*scaled[:,0] + w2*(scaled[:,1]-1) > 0).astype(int)
        return np.mean(np.abs(preds - iris_y))
    coords_x = []; coords_y = []; coords_z = []
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    for xi in xs:
        for yi in ys:
            coords_x.append(xi); coords_y.append(yi); coords_z.append(loss_fn(xi, yi))
    fig = go.FigureWidget(data=[go.Scatter3d(x=coords_x, y=coords_y, z=coords_z, mode='markers', marker=dict(size=2))])
    fig.update_layout(title="Loss landscape (discrete classifier)", width=900, height=600)
    display(fig)

# ---------------------------
# show_mlp (plot decision boundaries if file exists; otherwise show a placeholder)
# ---------------------------
def show_mlp(decision_boundary_file='decision_boundaries.txt'):
    if os.path.exists(decision_boundary_file):
        with open(decision_boundary_file,'r') as f:
            data = json.load(f)
    else:
        data = None

    X,y = make_moons(n_samples=200, noise=0.12)
    fig = go.FigureWidget()
    fig.add_scatter(x=X[y==0,0], y=X[y==0,1], mode='markers', marker=dict(color='blue'), name='class 0')
    fig.add_scatter(x=X[y==1,0], y=X[y==1,1], mode='markers', marker=dict(color='red'), name='class 1')
    if data is not None:
        # try to draw some boundaries (best-effort)
        try:
            db = np.array(data.get('relu', {}).get('3', {}).get('1', []))
            if db.size:
                fig.add_scatter(x=db[:,0], y=db[:,1], mode='markers', marker=dict(size=3, color='green'), name='db')
        except Exception:
            pass
    fig.update_layout(title="MLP / Decision boundaries (approx.)", width=800, height=500)
    display(fig)

# ---------------------------
# show_conv (image convolution demo)
# ---------------------------
def show_conv(image_path='taj_mahal.jpg'):
    # load sample image (if file missing, try skimage sample)
    try:
        im = io.imread(image_path, as_gray=True)
    except Exception:
        try:
            from skimage import data
            im = data.camera()
        except Exception:
            im = np.zeros((200,200))
    im = img_as_ubyte(im/np.max(im))
    # helper to convert to bytes for widget.Image
    import io as sysio
    from PIL import Image
    def arr2bytes(arr):
        im_pil = Image.fromarray(arr)
        buf = sysio.BytesIO()
        im_pil.save(buf, format='PNG')
        return buf.getvalue()

    img_widget = widgets.Image(value=arr2bytes(im), format='png', width=400, height=400)

    # kernel widgets
    kv = [widgets.FloatText(value=v, layout=widgets.Layout(width='40px')) for v in [-1,-1,-1,-1,8,-1,-1,-1,-1]]
    grid = widgets.GridBox(children=kv, layout=widgets.Layout(grid_template_columns='repeat(3, 40px)'))
    radio = widgets.RadioButtons(options=['Identity','Contrast','Edge','Vertical Edge','Horizontal Edge','Gaussian'], value='Identity')
    out = widgets.Output()

    def apply_kernel(change=None):
        k = np.array([float(kv[i].value) for i in range(9)]).reshape(3,3)
        kernel_const = 1
        if radio.value=='Identity':
            pass
        elif radio.value=='Gaussian':
            kernel_const = 1/16
        # convolve
        conv = ndimage.convolve(im.astype(float)/255.0, k, mode='mirror') * kernel_const
        conv_u = img_as_ubyte(np.clip(conv,0,1))
        img_widget.value = arr2bytes(conv_u)

    # attach observers
    for w in kv:
        w.observe(lambda ch: apply_kernel(), names='value')
    radio.observe(lambda ch: apply_kernel(), names='value')

    display(widgets.HBox([widgets.VBox([radio, grid]), img_widget]))

# ---------------------------
# show_dense (simple animation using frames)
# ---------------------------
def show_dense():
    # Build a stepped animation showing layers using a FigureWidget with many traces
    fig = go.FigureWidget()
    # static layout nodes positions (approx)
    xs = [0, 2, 2, 2, 5]; ys=[0,-3.5,0,3.5,0]
    fig.add_scatter(x=xs, y=ys, mode='markers+text', text=['in','h1','h2','h3','out'], textposition='top center', marker=dict(size=[30,30,30,30,30], color=['steelblue','red','red','red','green']))
    fig.update_layout(title="MLP schematic (animated)", width=900, height=450)
    display(fig)

    play = widgets.Play(min=0, max=9, value=0, interval=500)
    step = widgets.IntSlider(min=0, max=9, value=0)
    widgets.jslink((play,'value'), (step,'value'))

    def animate(frame):
        # This is just illustrative — adapt as needed
        with fig.batch_update():
            if frame==0:
                fig.data[0].marker.color = ['steelblue','red','red','red','green']
            elif frame==1:
                fig.data[0].marker.color = ['lightgray','red','red','red','green']
            elif frame==2:
                fig.data[0].marker.color = ['lightgray','lightgray','red','red','green']
            else:
                fig.data[0].marker.color = ['green']*5

    out = widgets.interactive_output(animate, {'frame': step})
    display(widgets.HBox([play, step, out]))

# ---------------------------
# show_gradient_descent (convex)
# ---------------------------
def show_gradient_descent():
    xs = np.linspace(-10,10,400)
    ys = xs**2 + 2
    fig = go.FigureWidget()
    fig.add_scatter(x=xs, y=ys, mode='lines')
    pt = go.Scatter(x=[-5], y=[(-5)**2 + 2], mode='markers', marker=dict(color='green', size=8))
    fig.add_trace(pt)
    fig.update_layout(title="Gradient descent on quadratic", width=800, height=450)
    display(fig)
    s_x0 = widgets.FloatSlider(min=-10,max=10,step=0.2,value=-5,description='x0')
    s_lr = widgets.BoundedFloatText(min=0.001,max=1.5,step=0.01,value=0.9,description='lr')
    play = widgets.Play(min=0,max=50,value=0,interval=100)
    step = widgets.IntSlider(min=0,max=50, value=0, description='step')
    widgets.jslink((play,'value'), (step,'value'))

    def update(x0, lr, stepv):
        xk = np.zeros(51); xk[0]=x0
        for k in range(1,51):
            xk[k] = xk[k-1] - lr*2*xk[k-1]
        with fig.batch_update():
            fig.data[1].x = xk[:stepv+1]; fig.data[1].y = xk[:stepv+1]**2 + 2
    out = widgets.interactive_output(update, {'x0': s_x0, 'lr': s_lr, 'stepv': step})
    display(widgets.VBox([widgets.HBox([s_x0, s_lr]), widgets.HBox([play, step]), out]))

# END OF FILE
