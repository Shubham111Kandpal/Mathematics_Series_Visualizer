import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fourier Series Visualizer", layout="wide")

st.title("🎵 Fourier Series Visualizer")
st.markdown("""
The Fourier Series represents a periodic function as an infinite sum of sines and cosines. It's a powerful tool in **signal processing**, **audio engineering**, and **solving PDEs**.

""")

st.latex(r"f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[a_n \cos(nx) + b_n \sin(nx)\right]")

# User controls
function_type = st.selectbox("Select waveform to approximate:", ["Square Wave", "Sawtooth Wave", "Triangle Wave"])
num_terms = st.slider("Number of Fourier Terms", min_value=1, max_value=100, value=10, step=1)
x_vals = np.linspace(-np.pi, np.pi, 1000)

# Original function definitions
def square_wave(x):
    return np.sign(np.sin(x))

def sawtooth_wave(x):
    return 2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))

def triangle_wave(x):
    return 2 * np.abs(2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))) - 1

# Fourier approximation builders
def fourier_square_wave(x, n_terms):
    result = np.zeros_like(x)
    for n in range(1, 2 * n_terms, 2):  # odd harmonics only
        result += (4 / (np.pi * n)) * np.sin(n * x)
    return result

def fourier_sawtooth_wave(x, n_terms):
    result = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        result += (-2 / (np.pi * n)) * np.sin(n * x)
    return result

def fourier_triangle_wave(x, n_terms):
    result = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        k = 2 * n - 1
        result += (8 / (np.pi**2 * k**2)) * (-1)**((n - 1)) * np.cos(k * x)
    return result

# Generate original and approximation
if function_type == "Square Wave":
    y_true = square_wave(x_vals)
    y_approx = fourier_square_wave(x_vals, num_terms)
elif function_type == "Sawtooth Wave":
    y_true = sawtooth_wave(x_vals)
    y_approx = fourier_sawtooth_wave(x_vals, num_terms)
else:
    y_true = triangle_wave(x_vals)
    y_approx = fourier_triangle_wave(x_vals, num_terms)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_true, mode='lines', name='Original Waveform', line=dict(color='black')))
fig.add_trace(go.Scatter(x=x_vals, y=y_approx, mode='lines', name=f'Fourier Approx (n={num_terms})', line=dict(color='red', dash='dot')))

fig.update_layout(title=f"Fourier Series Approximation of a {function_type}",
                  xaxis_title="x",
                  yaxis_title="f(x)",
                  legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# Explanation
with st.expander("📘 Real-life Significance of Fourier Series"):
    st.markdown("""
- Used in **audio/sound compression** (e.g., MP3, WAV).
- Essential in solving **heat/diffusion equations**.
- Forms the basis of **Fourier Transform**, used in **image processing**, **quantum mechanics**, and more.
- Breaks down any periodic function into its **harmonic components**.

Try increasing the number of terms to see how the approximation gets better!
    """)

st.markdown("---")
st.header("🔁 3D Visualization of Fourier Series Evolution")

# Parameters
n_terms = st.slider("Number of Fourier Terms", min_value=1, max_value=50, value=10)
x_vals = np.linspace(-np.pi, np.pi, 200)
X, N = np.meshgrid(x_vals, np.arange(1, n_terms + 1))

# Fourier series for f(x) = x on [-π, π] (odd function, sine series only)
def fourier_term(n, x):
    return (2 * ((-1)**(n + 1)) / n) * np.sin(n * x)

# Build series
Z = np.zeros_like(X)
for i in range(1, n_terms + 1):
    term = fourier_term(i, x_vals)
    Z[i-1, :] = Z[i-2, :] + term if i > 1 else term

# 3D Surface: X-axis (x), Y-axis (term index), Z-axis (cumulative sum)
fig = go.Figure(data=[
    go.Surface(
        x=x_vals,
        y=np.arange(1, n_terms + 1),
        z=Z,
        colorscale='Viridis'
    )
])
fig.update_layout(
    title="3D Evolution of Fourier Series for f(x) = x",
    scene=dict(
        xaxis_title="x (Input)",
        yaxis_title="Term Index n",
        zaxis_title="Fourier Approximation"
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

# Fourier Series: a_n = 1/n for odd n, 0 otherwise (square wave approximation)
st.header("📙 Fourier Series Spiral")
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
**Series:** $a_n = \begin{cases} \frac{1}{n}, & n \text{ odd} \\ 0, & n \text{ even} \end{cases}$  
**Description:** This spiral represents the Fourier coefficients for a square wave.  
- Terms alternate between non-zero and zero, showing harmonic decay.
- Only odd harmonics are present, visualized as jumps in spiral length.
- The pattern reflects wave reconstruction via frequency components.
""")

n_terms = st.slider("Number of terms", min_value=5, max_value=200, value=100)
fourier_series = np.array([1/n if n % 2 == 1 else 0 for n in range(1, n_terms + 1)])

fig = plot_spiral_series(fourier_series, title="Fourier Series Spiral", label_prefix="a")
# Create three columns and display the figure in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
with col2:
    st.pyplot(fig)
