import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Z-Series (Z-Transform) Visualizer", layout="wide")

st.title("🔁 Z-Series (Z-Transform) Visualizer")
st.markdown("""
The **Z-transform** is a powerful tool for analyzing discrete-time systems and signals. It’s the discrete analog of the **Laplace transform**, widely used in signal processing and control theory.

""")
st.latex(r"X(z) = \sum_{n=0}^{\infty} x[n] z^{-n}")

st.markdown("""
Where:
- \( x[n] \) is a discrete signal
- \( z \) is a complex variable
- This series converts a time-domain sequence into the **Z-domain** for easier manipulation and system analysis
""")

# Signal selection
signal_type = st.selectbox("Select a discrete-time signal:", ["Unit Impulse", "Unit Step", "Geometric Sequence", "Custom"])
length = st.slider("Length of the signal (n terms)", 5, 50, 20)

# Create signal
n_vals = np.arange(length)
if signal_type == "Unit Impulse":
    x_vals = np.zeros(length)
    x_vals[0] = 1
elif signal_type == "Unit Step":
    x_vals = np.ones(length)
elif signal_type == "Geometric Sequence":
    r = st.slider("Common ratio (r)", 0.1, 1.0, 0.5)
    x_vals = r**n_vals
elif signal_type == "Custom":
    st.info("Enter comma-separated values for x[n]:")
    user_input = st.text_input("x[n] =", "1, 0.5, 0.25, 0.125")
    try:
        x_vals = np.array([float(val.strip()) for val in user_input.split(",")])
        n_vals = np.arange(len(x_vals))
    except:
        st.error("Invalid input. Please enter comma-separated numeric values.")
        x_vals = np.zeros(length)

# Plot time-domain signal
st.subheader("🕒 Time-Domain Signal (x[n])")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=n_vals,
    y=x_vals,
    mode='markers+lines',
    name='x[n]',
    line=dict(dash='dot', color='royalblue'),
    marker=dict(size=8)
))

fig.update_layout(title=f"{signal_type} Signal",
                  xaxis_title="n",
                  yaxis_title="x[n]",
                  xaxis=dict(tickmode='linear'),
                  showlegend=False)

st.plotly_chart(fig, use_container_width=True)

fig, ax = plt.subplots()
ax.stem(n_vals, x_vals, basefmt=" ")
ax.set_xlabel("n")
ax.set_ylabel("x[n]")
ax.set_title(f"{signal_type} Signal")
st.pyplot(fig)

# Z-transform (symbolic)
z = sp.Symbol('z')
Z_transform = sum(x_vals[n] * z**(-n) for n in range(len(x_vals)))
Z_simplified = sp.simplify(Z_transform)

# Output
with st.expander("📜 Z-Transform Expression (Symbolic)"):
    st.latex(sp.latex(Z_simplified))

# Real-world section
with st.expander("🌍 Real-world Applications"):
    st.markdown("""
- **Signal Processing**: Filters, convolution, and spectral analysis of digital signals.
- **Control Systems**: Analysis of discrete feedback systems.
- **Communications**: Discrete-time encoding and decoding algorithms.
- **Finance**: Discrete-time models for forecasting and control.

The Z-transform simplifies complex difference equations into algebraic ones—just like Laplace does for continuous systems!
    """)

st.markdown("---")
st.header("🔁 3D Visualization of Z-Series (Z-Transform) Evolution")

# Inputs
n_terms = st.slider("Number of Terms (n)", min_value=5, max_value=50, value=20)
z_val = st.slider("z (Real Part)", min_value=0.1, max_value=3.0, value=1.5, step=0.1)

# Choose signal
signal_type = st.selectbox("Select x[n] (Discrete Signal)", ["Unit Step", "Exponential Decay", "Alternating", "Impulse"])

def get_signal(n):
    if signal_type == "Unit Step":
        return np.ones(n)
    elif signal_type == "Exponential Decay":
        return 0.9 ** np.arange(n)
    elif signal_type == "Alternating":
        return np.array([(-1)**i for i in range(n)])
    elif signal_type == "Impulse":
        sig = np.zeros(n)
        sig[0] = 1
        return sig

x_n = get_signal(n_terms)
n = np.arange(n_terms)
terms = x_n * z_val**(-n)
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
        name="Z-Series Terms"
    )
])
fig.update_layout(
    title=f"3D Evolution of Z-Series for {signal_type} with z = {z_val}",
    scene=dict(
        xaxis_title="n (Index)",
        yaxis_title="Term Value",
        zaxis_title="Partial Sum (Z-Transform)"
    )
)

st.plotly_chart(fig, use_container_width=True)

st.info("ℹ️ The Z-Transform is used to analyze discrete-time systems in the frequency domain.")

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

# Z-Series: x[n] = a^n * u[n], example with a = 0.7 and u[n] = 1 (unit step)
st.header("📙 Z-Series Spiral")
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
**Series:** $x[n] = a^n \cdot u[n]$ (Z-Transform input, with $a = 0.7$ and $u[n] = 1$)  
**Description:** This spiral represents a causal discrete-time signal for the Z-Transform.  
- The signal decays exponentially due to $a^n$ where $|a| < 1$.
- The spiral simulates the pole structure and time evolution of the system.
- Common in digital signal processing, this shape reflects system stability and behavior.
""")

n_terms = st.slider("Number of terms", min_value=5, max_value=100, value=40)
a = 0.7
z_series = a ** np.arange(n_terms)  # Unit step makes all terms active

fig = plot_spiral_series(z_series, title="Z-Series Spiral", label_prefix="x")
# Create three columns and display the figure in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
with col2:
    st.pyplot(fig)
