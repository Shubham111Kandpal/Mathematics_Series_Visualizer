import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Binomial Series Visualizer", layout="wide")

st.title("🔢 Binomial Series Visualizer")
st.markdown("""
The **Binomial Series** is an expansion of \( (1 + x)^k \), valid even when the exponent \( k \) is not a whole number.

""")
st.latex(r"(1 + x)^k = \sum_{n=0}^{\infty} \binom{k}{n} x^n = 1 + kx + \frac{k(k - 1)}{2!}x^2 + \frac{k(k - 1)(k - 2)}{3!}x^3 + \dots")

st.markdown("""
This expansion works for any real \( k \) as long as \( |x| < 1 \).  
It's useful for approximations in calculus, physics, and numerical methods.
""")

# Inputs
k = st.number_input("Enter exponent k (can be fractional)", value=0.5, step=0.1)
n_terms = st.slider("Number of terms to include", min_value=1, max_value=30, value=5)
x_range = st.slider("X-Axis Range (zoom in for |x| < 1)", -5.0, 5.0, (-1.5, 1.5))

x_vals = np.linspace(x_range[0], x_range[1], 400)

# Symbolic series expansion
x = sp.Symbol('x')
binomial_expansion = sp.series((1 + x)**k, x, 0, n_terms + 1).removeO()
f_series = sp.lambdify(x, binomial_expansion, modules=["numpy"])
f_exact = sp.lambdify(x, (1 + x)**k, modules=["numpy"])

# Evaluate functions
y_approx = f_series(x_vals)
y_true = f_exact(x_vals)

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_true, name="(1 + x)^k", line=dict(color='green')))
fig.add_trace(go.Scatter(x=x_vals, y=y_approx, name=f"Binomial Approx (n={n_terms})", line=dict(color='purple', dash='dot')))

fig.update_layout(title=f"Binomial Series Approximation for (1 + x)^{k}",
                  xaxis_title="x", yaxis_title="y",
                  legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# Formula viewer
with st.expander("📜 View Expansion Formula"):
    st.latex(sp.latex(binomial_expansion))

# Real-world context
with st.expander("🌍 Real-world Applications"):
    st.markdown("""
- **Numerical methods**: Approximating functions like \( \sqrt{1+x} \), \( (1+x)^k \), or reciprocals.
- **Physics**: Binomial expansion is used to simplify equations of motion under approximations.
- **Engineering**: Control theory, signal processing, and error estimation.
- **Finance**: Option pricing models like Black-Scholes use binomial tree approximations.
    """)

from scipy.special import comb

st.markdown("---")
st.header("🔁 3D Visualization of Binomial Series Evolution")

# Inputs
n_terms = st.slider("Number of Terms (n)", min_value=5, max_value=50, value=20)
x_val = st.slider("Value of x", min_value=-1.0, max_value=1.0, value=0.5, step=0.05)
k = st.slider("Power (k)", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)

# Compute binomial terms and partial sums
n = np.arange(n_terms)
terms = np.array([comb(k, i) * x_val**i for i in n])
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
        name="Binomial Terms"
    )
])
fig.update_layout(
    title=f"3D Evolution of Binomial Series for (1 + x)^{k}",
    scene=dict(
        xaxis_title="Term Index n",
        yaxis_title="Term Value",
        zaxis_title="Partial Sum"
    )
)

st.plotly_chart(fig, use_container_width=True)

# Display theoretical value
try:
    exact = (1 + x_val) ** k
    st.success(f"✅ Theoretical value: (1 + {x_val})^{k} = {exact:.5f}")
except:
    st.warning("⚠️ Could not compute the theoretical value.")

from math import comb

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

# Binomial Series: (1 + x)^k where k is any real number (e.g., 0.5), a_n = C(k, n) * x^n
st.header("📗 Binomial Series Spiral")
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
**Series:** $a_n = \binom{k}{n} x^n$ (for $k = 0.5$, $x = 0.5$)  
**Description:** This spiral represents the generalized binomial expansion of $(1 + x)^k$.  
- Coefficients are computed using the extended binomial theorem.
- The values slowly decrease in magnitude depending on $x$ and $k$.
- This spiral captures how fractional exponents still allow infinite series expansions.
""")

n_terms = st.slider("Number of terms", min_value=5, max_value=100, value=30)
k = 0.5
x_val = 0.5
binomial_series = [np.prod([(k - j)/ (j + 1) for j in range(n)]) * x_val**n for n in range(n_terms)]

fig = plot_spiral_series(binomial_series, title="Binomial Series Spiral", label_prefix="a")
# Create three columns and display the figure in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
with col2:
    st.pyplot(fig)
