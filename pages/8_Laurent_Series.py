import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Laurent Series Visualizer", layout="wide")

st.title("🔄 Laurent Series Visualizer")
st.markdown("""
The **Laurent Series** is a representation of a complex function that includes both positive and negative powers of \( (x - a) \). It's particularly useful near **singularities** where Taylor series fail.

""")
st.latex(r"f(x) = \sum_{n=-\infty}^{\infty} c_n (x - a)^n")

st.markdown("""
This makes it valuable in **complex analysis**, **residue calculus**, and **electrical engineering**.
""")

# Select function
function_choice = st.selectbox("Choose a function to expand into Laurent Series:", [
    "1/x",
    "1/(1 - x)",
    "1/(x^2 + 1)",
    "exp(1/x)"
])

# Input: expansion center and number of terms
a = st.number_input("Expansion point a (around which the series is built)", value=0.0)
n_terms = st.slider("Number of terms on each side (positive & negative)", 1, 20, 5)
x_range = st.slider("X-axis range", -5.0, 5.0, (-2.0, 2.0))

# Define x
x = sp.Symbol('x')
x_vals = np.linspace(x_range[0], x_range[1], 600)

# Choose symbolic function
if function_choice == "1/x":
    func = 1 / x
elif function_choice == "1/(1 - x)":
    func = 1 / (1 - x)
elif function_choice == "1/(x^2 + 1)":
    func = 1 / (x**2 + 1)
elif function_choice == "exp(1/x)":
    func = sp.exp(1 / x)

# Attempt Laurent expansion using SymPy's series around x=a
try:
    laurent_series = func.series(x, a, n=n_terms * 2 + 1).removeO()
    f_series = sp.lambdify(x, laurent_series, modules=["numpy"])
    y_approx = f_series(x_vals)
except Exception as e:
    laurent_series = None
    y_approx = None

# Compute true function for comparison
f_true = sp.lambdify(x, func, modules=["numpy"])
try:
    y_true = f_true(x_vals)
except Exception:
    y_true = np.full_like(x_vals, np.nan)

# Filter singularities from x_vals (e.g., x=0 for 1/x)
with np.errstate(divide='ignore', invalid='ignore'):
    y_true = np.where(np.abs(y_true) > 1000, np.nan, y_true)
    if y_approx is not None:
        y_approx = np.where(np.abs(y_approx) > 1000, np.nan, y_approx)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_true, name="Actual Function", line=dict(color='green')))
if y_approx is not None:
    fig.add_trace(go.Scatter(x=x_vals, y=y_approx, name=f"Laurent Series (±{n_terms} terms)", line=dict(color='red', dash='dot')))

fig.update_layout(title=f"Laurent Series Approximation of {function_choice}",
                  xaxis_title="x", yaxis_title="f(x)",
                  legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# Formula
with st.expander("📜 View Laurent Series Expression"):
    if laurent_series is not None:
        st.latex(sp.latex(laurent_series))
    else:
        st.error("Could not compute Laurent series for the selected function.")

# Context
with st.expander("🌍 Real-world Applications"):
    st.markdown("""
- **Complex analysis**: Key tool for studying behavior near poles and essential singularities.
- **Residue calculus**: Helps compute integrals in the complex plane.
- **Engineering**: Used in electrical circuit analysis involving impedances and inverse Laplace transforms.
- **Physics**: Laurent series appear in quantum field theory and wave equations near singularities.
    """)

st.markdown("---")
st.header("🔁 3D Visualization of Laurent Series Evolution")

# Inputs
total_terms = st.slider("Total Number of Terms (odd number)", min_value=5, max_value=51, value=21, step=2)
x_val = st.slider("Value of x", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
a_val = st.slider("Expansion Point (a)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)

# Term indices: balanced around 0 (e.g., -5 to 5)
half = total_terms // 2
n = np.arange(-half, half + 1)

# Define coefficients — for simplicity, c_n = 1 / (|n| + 1)
c_n = np.array([1 / (abs(i) + 1) for i in n])

# Compute terms and partial sums
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
        name="Laurent Terms"
    )
])
fig.update_layout(
    title="3D Evolution of Laurent Series (Simulated)",
    scene=dict(
        xaxis_title="Exponent n",
        yaxis_title="Term Value",
        zaxis_title="Partial Sum"
    )
)

st.plotly_chart(fig, use_container_width=True)

st.info("ℹ️ This simulated Laurent series includes both positive and negative powers. It’s often used near singularities where Taylor series fails.")

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

# Laurent Series: includes negative powers, e.g., a_n = 1/n for n = -5 to 5 excluding 0
st.header("📗 Laurent Series Spiral")
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
**Series:** $a_n = \frac{1}{n}$ for $n = -5$ to $-1$ and $1$ to $5$ (excluding $n=0$)  
**Description:** This spiral reflects both negative and positive powers of a Laurent Series.  
- Negative powers are visualized in reverse, showing behavior near singularities.
- Positive powers behave like a regular power or harmonic series.
- This dual nature creates a balanced spiral pattern around the origin.
""")

n_range = 5
positive_terms = [1/n for n in range(1, n_range + 1)]
negative_terms = [1/n for n in range(-n_range, 0)]
laurent_series = negative_terms + positive_terms

fig = plot_spiral_series(laurent_series, title="Laurent Series Spiral", label_prefix="a")
# Create three columns and display the figure in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
with col2:
    st.pyplot(fig)
