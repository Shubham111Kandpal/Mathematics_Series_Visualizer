import streamlit as st
import numpy as np
import sympy as sp
from scipy.special import legendre
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Legendre Series Visualizer", layout="wide")

st.title("🔷 Legendre Series Visualizer")
st.markdown("""
The **Legendre Series** represents a function as a sum of **Legendre polynomials**, which are orthogonal on the interval \([-1, 1]\).

""")
st.latex(r"f(x) = \sum_{n=0}^{\infty} a_n P_n(x)")

st.markdown("""
Where:
- \( P_n(x) \) is the Legendre polynomial of degree \( n \)
- \( a_n = \frac{2n + 1}{2} \int_{-1}^{1} f(x) P_n(x)\, dx \)

This expansion is used in **spherical problems** in physics, like gravitational or electric potential around spheres.
""")

# Function choice
function_choice = st.selectbox("Function to approximate:", [
    "x^2", "|x|", "cos(x)", "exp(x)"
])
num_terms = st.slider("Number of Legendre terms", 1, 20, 5)

# Define x
x_vals = np.linspace(-1, 1, 500)

# Define the target function
if function_choice == "x^2":
    f = lambda x: x**2
elif function_choice == "|x|":
    f = lambda x: np.abs(x)
elif function_choice == "cos(x)":
    f = lambda x: np.cos(x)
elif function_choice == "exp(x)":
    f = lambda x: np.exp(x)

# Compute coefficients a_n
def legendre_series_coefficients(f, n_terms):
    coeffs = []
    for n in range(n_terms):
        Pn = legendre(n)
        integrand = lambda x: f(x) * Pn(x)
        x_int = np.linspace(-1, 1, 1000)
        y_int = integrand(x_int)
        a_n = (2 * n + 1) / 2 * np.trapz(y_int, x_int)
        coeffs.append(a_n)
    return coeffs

# Build the approximation
def evaluate_legendre_series(x, coeffs):
    result = np.zeros_like(x)
    for n, a_n in enumerate(coeffs):
        result += a_n * legendre(n)(x)
    return result

# Compute values
coeffs = legendre_series_coefficients(f, num_terms)
y_true = f(x_vals)
y_approx = evaluate_legendre_series(x_vals, coeffs)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_true, name="Original Function", line=dict(color='green')))
fig.add_trace(go.Scatter(x=x_vals, y=y_approx, name=f"Legendre Approx (n={num_terms})", line=dict(color='purple', dash='dot')))

fig.update_layout(title=f"Legendre Series Approximation of {function_choice}",
                  xaxis_title="x", yaxis_title="f(x)",
                  legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# Explanation
with st.expander("📜 What are Legendre Polynomials?"):
    st.markdown("""
Legendre polynomials \( P_n(x) \) are solutions to **Legendre’s differential equation** and are orthogonal over \([-1, 1]\):

""")
    st.latex(r"\int_{-1}^{1} P_m(x) P_n(x)\, dx = 0 \quad \text{for } m \ne n")

    st.markdown("""
They appear in:
- Spherical harmonics in physics
- Solving Laplace’s equation in spherical coordinates
- Approximation of complex functions on bounded domains
    """)

# Use case
with st.expander("🌍 Real-world Applications"):
    st.markdown("""
- **Quantum mechanics**: Wavefunctions of electrons in atoms
- **Geophysics & astronomy**: Modeling gravity and electric fields
- **Computer graphics**: Spherical harmonics lighting
- **Approximation theory**: Representing functions on bounded intervals
    """)

st.markdown("---")
st.header("🔁 3D Visualization of Legendre Series Evolution")

# Inputs
n_terms = st.slider("Number of Terms (n)", min_value=2, max_value=50, value=10)
x_vals = np.linspace(-1, 1, 200)

# Choose target function
target_func_name = st.selectbox("Function to Approximate", ["x", "sin(πx)", "x²"])
if target_func_name == "x":
    f_target = lambda x: x
elif target_func_name == "sin(πx)":
    f_target = lambda x: np.sin(np.pi * x)
elif target_func_name == "x²":
    f_target = lambda x: x**2

# Compute Legendre coefficients via projection
def legendre_coeff(f, n, x_vals):
    Pn = legendre(n)
    Pn_vals = Pn(x_vals)
    weight = 1  # Weight for standard Legendre polynomials is 1
    return np.trapz(f(x_vals) * Pn_vals, x_vals) * (2 * n + 1) / 2

coeffs = np.array([legendre_coeff(f_target, n, x_vals) for n in range(n_terms)])

# Build series
Y = np.zeros((n_terms, len(x_vals)))
for n in range(n_terms):
    Pn_vals = legendre(n)(x_vals)
    Y[n, :] = Y[n-1, :] + coeffs[n] * Pn_vals if n > 0 else coeffs[0] * Pn_vals

# 3D Surface
fig = go.Figure(data=[
    go.Surface(
        x=x_vals,
        y=np.arange(1, n_terms + 1),
        z=Y,
        colorscale="Viridis"
    )
])
fig.update_layout(
    title=f"3D Evolution of Legendre Series for {target_func_name}",
    scene=dict(
        xaxis_title="x",
        yaxis_title="Term Index n",
        zaxis_title="Approximation"
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

# Legendre Series: Coefficients of Legendre polynomials evaluated at x = 0.5
st.header("📘 Legendre Series Spiral")
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
**Series:** $a_n = P_n(0.5)$ where $P_n$ is the $n^{th}$ Legendre polynomial  
**Description:** This spiral shows the coefficients of the Legendre polynomials evaluated at a fixed point.  
- The terms oscillate and gradually shrink for larger $n$.
- This captures the orthogonal nature of the Legendre basis functions.
- The pattern reflects approximations of functions over $[-1, 1]$ using Legendre expansion.
""")

n_terms = st.slider("Number of terms", min_value=5, max_value=100, value=30)
x_val = 0.5
legendre_series = [legendre(n)(x_val) for n in range(n_terms)]

fig = plot_spiral_series(legendre_series, title="Legendre Series Spiral", label_prefix="a")
# Create three columns and display the figure in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
with col2:
    st.pyplot(fig)
