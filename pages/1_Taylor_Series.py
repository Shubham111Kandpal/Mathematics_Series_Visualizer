import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import math
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Taylor Series Visualizer", layout="wide")

st.title("📐 Taylor Series Visualizer")
st.markdown("""
The **Taylor Series** allows us to approximate a function as an infinite sum of its derivatives around a point \( a \):

""")
st.latex(r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x - a)^n")

st.markdown("""
It's widely used in calculators, computers, physics simulations, and numerical analysis.
""")

# User inputs
function_choice = st.selectbox("Choose a function to approximate:", ["sin(x)", "cos(x)", "e^x", "ln(1+x)", "arctan(x)"])
a = st.number_input("Expansion point (a):", value=0.0)
n_terms = st.slider("Number of terms (n)", 1, 30, 10)
x_range = st.slider("X-axis range:", -10.0, 10.0, (-5.0, 5.0))

# Symbolic math setup
x = sp.Symbol('x')
if function_choice == "sin(x)":
    func = sp.sin(x)
elif function_choice == "cos(x)":
    func = sp.cos(x)
elif function_choice == "e^x":
    func = sp.exp(x)
elif function_choice == "ln(1+x)":
    func = sp.ln(1 + x)
elif function_choice == "arctan(x)":
    func = sp.atan(x)

# Taylor expansion at point a
taylor_series = func.series(x, a, n=n_terms + 1).removeO()

# Numerical conversion
f_exact = sp.lambdify(x, func, modules=["numpy"])
f_approx = sp.lambdify(x, taylor_series, modules=["numpy"])

x_vals = np.linspace(x_range[0], x_range[1], 400)
y_true = f_exact(x_vals)
y_taylor = f_approx(x_vals)

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_true, name=f"{function_choice}", line=dict(color='green')))
fig.add_trace(go.Scatter(x=x_vals, y=y_taylor, name=f"Taylor Approx. (n={n_terms})", line=dict(color='red', dash='dot')))

fig.update_layout(title=f"Taylor Series Approximation of {function_choice}",
                  xaxis_title="x", yaxis_title="f(x)",
                  legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# LaTeX Series
with st.expander("📜 View Taylor Series Expression"):
    st.latex(sp.latex(taylor_series))

# Applications
with st.expander("🌍 Real-world Applications"):
    st.markdown("""
- **Scientific computing**: Used by calculators and computers to evaluate functions quickly.
- **Physics**: Approximating motion, energy, and wave equations near a known state.
- **Engineering**: Control systems and simulations often use Taylor-based approximations.
- **Machine Learning**: Used in optimization algorithms and approximation theory.
    """)

# === 3-D
st.markdown("---")
st.header("🔁 3D Visualization of Taylor Series Evolution")

# Inputs
n_terms = st.slider("Number of Terms (n)", min_value=5, max_value=50, value=20)
x_val = st.slider("Value of x", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
a_val = st.slider("Expansion Point a", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

# Select function
function_options = {
    "e^x": sp.exp(sp.Symbol('x')),
    "sin(x)": sp.sin(sp.Symbol('x')),
    "cos(x)": sp.cos(sp.Symbol('x')),
    "ln(1 + x)": sp.ln(1 + sp.Symbol('x')),
    "1 / (1 - x)": 1 / (1 - sp.Symbol('x'))
}
func_name = st.selectbox("Select a function for Taylor Expansion", list(function_options.keys()))
f_expr = function_options[func_name]

# Symbol setup
x = sp.Symbol('x')
f_derivatives = [f_expr]
for i in range(1, n_terms):
    f_derivatives.append(sp.diff(f_derivatives[i-1], x))

# Compute term values at 'a' and evaluate Taylor terms at 'x_val'
terms = []
for n, deriv in enumerate(f_derivatives):
    deriv_at_a = deriv.subs(x, a_val)
    term_val = float(deriv_at_a) / math.factorial(n) * (x_val - a_val)**n
    terms.append(term_val)

terms = np.array(terms)
partial_sums = np.cumsum(terms)
n = np.arange(n_terms)

# Plot
fig = go.Figure(data=[
    go.Scatter3d(
        x=n,
        y=terms,
        z=partial_sums,
        mode='lines+markers',
        marker=dict(size=4),
        line=dict(width=4),
        name="Taylor Terms"
    )
])

fig.update_layout(
    title=f"3D Evolution of Taylor Series for {func_name} at x = {x_val}",
    scene=dict(
        xaxis_title="Term Index n",
        yaxis_title="Term Value",
        zaxis_title="Partial Sum"
    )
)

st.plotly_chart(fig, use_container_width=True)

def plot_spiral_series(series, title="Spiral Series Representation", label_prefix="Term", num_annotate=10):
    n_terms = len(series)
    theta = np.linspace(0, 4 * np.pi, n_terms)
    r = series

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, marker='o', markersize=3, linewidth=1.5, label=title)

    for i in range(0, n_terms, num_annotate):
        ax.text(x[i], y[i], f"{label_prefix}{i}", fontsize=8, color='blue')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    
    return fig

# Taylor Series for e^x at x = 1: a_n = 1/n!
st.header("📗 Taylor Series Spiral")
with st.expander("ℹ️ What do the X and Y axes represent?"):
    st.markdown(r"""
Each point on the spiral is plotted using polar coordinates:
- **Angle (θ)** increases with the term index.
- **Radius (r)** is the value of the term.

These are converted to Cartesian coordinates:
- $x = r \cdot \cos(\theta)$  
- $y = r \cdot \sin(\theta)$

The **X and Y axes** represent spatial positions showing how the series evolves.  
They do **not** directly map to time or frequency — rather, they form a visual path that encodes magnitude and order.
""")
st.markdown(r"""
**Series:** $a_n = \frac{1}{n!}$ (Taylor expansion of $e^x$ at $x=1$)  
**Description:** This spiral visualizes the decreasing terms of the Taylor Series for $e^x$.  
- The terms shrink rapidly due to factorial growth in the denominator.
- This creates a tightly winding spiral toward the center.
- It reflects the fast convergence behavior of the exponential Taylor series.
""")

n_terms = st.slider("Number of terms", min_value=5, max_value=50, value=20)
taylor_series = 1 / np.array([math.factorial(n) for n in range(n_terms)])

fig = plot_spiral_series(taylor_series, title="Taylor Series Spiral", label_prefix="a")
# Create three columns and display the figure in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
with col2:
    st.pyplot(fig)

