import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Power Series Visualizer", layout="wide")

st.title("💡 Power Series Visualizer")
st.markdown("""
A **Power Series** is an infinite series of the form:

""")
st.latex(r"\sum_{n=0}^{\infty} c_n (x - a)^n")

st.markdown("""
Where:
- \( c_n \) are the coefficients
- \( a \) is the center of the expansion
- It generalizes many other series like Taylor, Maclaurin, Binomial

We’ll explore common power series representations of well-known functions.
""")

# Function selection
function_choice = st.selectbox("Select a function to approximate using its power series:", [
    "e^x",
    "sin(x)",
    "cos(x)",
    "ln(1 + x)",
    "arctan(x)"
])

n_terms = st.slider("Number of terms in the power series", 1, 40, 10)
x_range = st.slider("X-Axis Range", -5.0, 5.0, (-2.0, 2.0))

x_vals = np.linspace(x_range[0], x_range[1], 400)

# Define symbolic x
x = sp.Symbol('x')

# Predefined series expressions
if function_choice == "e^x":
    func = sp.exp(x)
elif function_choice == "sin(x)":
    func = sp.sin(x)
elif function_choice == "cos(x)":
    func = sp.cos(x)
elif function_choice == "ln(1 + x)":
    func = sp.ln(1 + x)
elif function_choice == "arctan(x)":
    func = sp.atan(x)

# Build power series around a = 0
power_series = func.series(x, 0, n_terms + 1).removeO()
f_approx = sp.lambdify(x, power_series, modules=["numpy"])
f_exact = sp.lambdify(x, func, modules=["numpy"])

y_approx = f_approx(x_vals)
y_true = f_exact(x_vals)

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_true, name=f"{function_choice}", line=dict(color='green')))
fig.add_trace(go.Scatter(x=x_vals, y=y_approx, name=f"Power Series Approx (n={n_terms})", line=dict(color='blue', dash='dot')))

fig.update_layout(title=f"Power Series Approximation of {function_choice}",
                  xaxis_title="x", yaxis_title="f(x)",
                  legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# Formula
with st.expander("📜 View Power Series Expression"):
    st.latex(sp.latex(power_series))

# Use cases
with st.expander("🌍 Real-world Applications"):
    st.markdown("""
- **Numerical computation**: Approximating complex functions using basic polynomials
- **Physics**: Used in wave functions, quantum mechanics, and motion analysis
- **Machine learning**: Kernel expansions and function approximation techniques
- **Engineering**: Control systems, stability analysis, and signal modeling
    """)

st.markdown("---")
st.header("🔁 3D Visualization of Power Series Evolution")

# Inputs
n_terms = st.slider("Number of Terms (n)", min_value=5, max_value=50, value=20)
x_val = st.slider("Value of x", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
a_val = st.slider("Center (a)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

# Choose coefficient pattern
pattern = st.selectbox("Coefficient Pattern (cₙ)", ["All 1s", "Alternating ±1", "Decreasing 1/n"])

def get_coefficients(n_terms, pattern):
    if pattern == "All 1s":
        return np.ones(n_terms)
    elif pattern == "Alternating ±1":
        return np.array([(-1)**n for n in range(n_terms)])
    elif pattern == "Decreasing 1/n":
        return np.array([1 / (n+1) for n in range(n_terms)])

c_n = get_coefficients(n_terms, pattern)

# Compute terms and partial sums
n = np.arange(n_terms)
terms = c_n * (x_val - a_val)**n
partial_sums = np.cumsum(terms)

# 3D Plot
fig = go.Figure(data=[
    go.Scatter3d(
        x=n,
        y=terms,
        z=partial_sums,
        mode='lines+markers',
        marker=dict(size=4),
        line=dict(width=4),
        name="Power Series Terms"
    )
])
fig.update_layout(
    title="3D Evolution of Power Series",
    scene=dict(
        xaxis_title="Term Index n",
        yaxis_title="Term Value",
        zaxis_title="Partial Sum"
    )
)

st.plotly_chart(fig, use_container_width=True)

def plot_spiral_series(series, title="Spiral Series Representation", label_prefix="Term", num_annotate=10):
    n_terms = len(series)
    theta = np.linspace(0, 6 * np.pi, n_terms)
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

# Power Series: a_n = x^n, where |x| < 1 for convergence (e.g., x = 0.8)
st.header("📙 Power Series Spiral")
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
**Series:** $a_n = x^n$ for $|x| < 1$ (example: $x = 0.8$)  
**Description:** This spiral visualizes a basic power series where each term is a power of $x$.  
- The values decrease geometrically for $|x| < 1$, showing convergence.
- The spiral curls inward with a smooth and regular pattern.
- This plot represents the foundation of many other series like Taylor and Laurent.
""")

n_terms = st.slider("Number of terms", min_value=5, max_value=100, value=40)
x_val = 0.8
power_series = x_val ** np.arange(n_terms)

fig = plot_spiral_series(power_series, title="Power Series Spiral", label_prefix="a")
# Create three columns and display the figure in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
with col2:
    st.pyplot(fig)
