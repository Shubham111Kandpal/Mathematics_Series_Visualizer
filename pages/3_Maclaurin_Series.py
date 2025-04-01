import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import math
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Maclaurin Series Visualizer", layout="wide")

st.title("📉 Maclaurin Series Visualizer")
st.markdown("""
The **Maclaurin Series** is a special case of the Taylor Series, expanded about the point \( a = 0 \). It's used to approximate functions with polynomials around zero.

""")

st.latex(r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!} x^n")

# Select function to approximate
function_choice = st.selectbox("Select a function to approximate:", ["e^x", "sin(x)", "cos(x)", "ln(1 + x)", "1 / (1 - x)"])
num_terms = st.slider("Number of Maclaurin Terms", min_value=1, max_value=30, value=5, step=1)
x_range = st.slider("X-Axis Range", -5.0, 5.0, (-2.0, 2.0))
x_vals = np.linspace(x_range[0], x_range[1], 400)

# SymPy expressions
x = sp.Symbol('x')
if function_choice == "e^x":
    func = sp.exp(x)
elif function_choice == "sin(x)":
    func = sp.sin(x)
elif function_choice == "cos(x)":
    func = sp.cos(x)
elif function_choice == "ln(1 + x)":
    func = sp.ln(1 + x)
elif function_choice == "1 / (1 - x)":
    func = 1 / (1 - x)

# Compute Maclaurin Series
maclaurin_series = func.series(x, 0, num_terms + 1).removeO()
f_lambdified = sp.lambdify(x, maclaurin_series, modules=["numpy"])
f_true = sp.lambdify(x, func, modules=["numpy"])

# Get values
y_approx = f_lambdified(x_vals)
y_true = f_true(x_vals)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_true, name="Actual Function", line=dict(color="green")))
fig.add_trace(go.Scatter(x=x_vals, y=y_approx, name=f"Maclaurin Approx (n={num_terms})", line=dict(color="orange", dash='dot')))

fig.update_layout(title=f"Maclaurin Series Approximation of {function_choice}",
                  xaxis_title="x", yaxis_title="f(x)", legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# Show symbolic formula
with st.expander("📜 View Maclaurin Series Formula"):
    st.latex(fr"{sp.latex(maclaurin_series)}")

# Use-case explanation
with st.expander("🌍 Real-world Applications"):
    st.markdown(f"""
- Approximating functions like `{function_choice}` near zero in scientific computing.
- Used in computer algebra systems, numerical integration, and symbolic computation.
- Essential in **engineering**, **physics**, and **machine learning** (e.g., activation function approximations).
    """)

st.markdown("---")
st.header("🔁 3D Visualization of Maclaurin Series Evolution")

# Controls
n_terms = st.slider("Number of Terms (n)", min_value=5, max_value=50, value=20)
x_val = st.slider("Value of x", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)

# Function options
function_options = {
    "e^x": sp.exp(sp.Symbol('x')),
    "sin(x)": sp.sin(sp.Symbol('x')),
    "cos(x)": sp.cos(sp.Symbol('x')),
    "ln(1 + x)": sp.ln(1 + sp.Symbol('x')),
    "1 / (1 - x)": 1 / (1 - sp.Symbol('x'))
}
func_name = st.selectbox("Select a function for Maclaurin Expansion", list(function_options.keys()))
f_expr = function_options[func_name]

# Symbolic derivatives at a = 0
x = sp.Symbol('x')
f_derivatives = [f_expr]
for i in range(1, n_terms):
    f_derivatives.append(sp.diff(f_derivatives[i-1], x))

terms = []
for n, deriv in enumerate(f_derivatives):
    deriv_at_0 = deriv.subs(x, 0)
    term_val = float(deriv_at_0) / math.factorial(n) * x_val**n
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
        name="Maclaurin Terms"
    )
])

fig.update_layout(
    title=f"3D Evolution of Maclaurin Series for {func_name} at x = {x_val}",
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

# Maclaurin Series for sin(x) at x = π/4: a_n = (-1)^n * (π/4)^(2n+1) / (2n+1)!
st.header("📘 Maclaurin Series Spiral")
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
**Series:** $a_n = \frac{(-1)^n (\pi/4)^{2n+1}}{(2n+1)!}$ (Maclaurin series for $\sin(x)$ at $x = \frac{\pi}{4}$)  
**Description:** This spiral shows the alternating nature of the Maclaurin Series for $\sin(x)$.  
- The terms alternate in sign, producing an in-and-out spiral.
- The magnitudes decay due to factorials in the denominator.
- The spiral reflects the smooth convergence of the sine approximation.
""")

n_terms = st.slider("Number of terms", min_value=5, max_value=100, value=30)
x_val = np.pi / 4
try:
    maclaurin_series = np.array([
        ((-1)**n * x_val**(2*n + 1)) / math.factorial(2*n + 1)
        for n in range(n_terms)
    ])
    fig = plot_spiral_series(maclaurin_series, title="Maclaurin Series Spiral", label_prefix="a")
    # Create three columns and display the figure in the center column
    col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
    with col2:
        st.pyplot(fig)

except OverflowError:
    st.error("⚠️ OverflowError: int too large to convert to float.\n\nTry reducing the number of terms.")

