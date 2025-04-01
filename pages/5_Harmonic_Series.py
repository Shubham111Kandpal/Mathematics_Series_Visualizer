import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Harmonic Series Visualizer", layout="wide")

st.title("📈 Harmonic Series Visualizer")
st.markdown("""
The **Harmonic Series** is defined as:

""")
st.latex(r"\sum_{n=1}^{\infty} \frac{1}{n} = 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \dots")

st.markdown("""
Despite its terms getting smaller and smaller, the harmonic series **diverges**—meaning its partial sums grow without bound, but very slowly.

It's a great example that **"small terms" don't always mean convergence**.
""")

# User input
n_terms = st.slider("Number of terms to plot", 1, 1000, 100, step=10)

# Generate terms and partial sums
n = np.arange(1, n_terms + 1)
terms = 1 / n
partial_sums = np.cumsum(terms)

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=n, y=terms, mode='markers', name='Terms (1/n)', marker=dict(color='orange')))
fig.add_trace(go.Scatter(x=n, y=partial_sums, mode='lines+markers', name='Partial Sum', line=dict(color='blue')))

fig.update_layout(title="Harmonic Series: Terms and Partial Sums",
                  xaxis_title="n", yaxis_title="Value",
                  legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# Info
with st.expander("🧠 Why the Harmonic Series Diverges"):
    st.markdown("""
Even though the terms shrink toward 0, the harmonic series diverges.  
A famous proof involves **grouping terms**:
""")
    st.latex(r"""
\frac{1}{2} + \left(\frac{1}{3} + \frac{1}{4}\right) + \left(\frac{1}{5} + \frac{1}{6} + \frac{1}{7} + \frac{1}{8}\right) + \dots
""")
    st.markdown("Each group adds up to at least 0.5, and since there are infinitely many such groups, the series diverges!")

with st.expander("🌍 Real-world Applications"):
    st.markdown("""
- **Data science**: Appears in analysis of algorithms (e.g., time complexity of QuickSort).
- **Physics**: Occurs in wave interference and energy distribution.
- **Mathematics**: Benchmark for comparison tests in convergence analysis.
    """)

st.markdown("---")
st.header("🔁 3D Visualization of Harmonic Series Evolution")

# Inputs
n_terms = st.slider("Number of Terms (n)", min_value=5, max_value=100, value=30)

# Compute harmonic terms
n = np.arange(1, n_terms + 1)
terms = 1 / n
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
        name="Harmonic Terms"
    )
])
fig.update_layout(
    title="3D Evolution of Harmonic Series",
    scene=dict(
        xaxis_title="Term Index n",
        yaxis_title="Term Value (1/n)",
        zaxis_title="Partial Sum"
    )
)

st.plotly_chart(fig, use_container_width=True)

# Informational message
st.info("ℹ️ The Harmonic Series diverges — the partial sum grows without bound, though very slowly.")

def plot_spiral_series(series, title="Spiral Series Representation", label_prefix="Term", num_annotate=10):
    n_terms = len(series)
    theta = np.linspace(0, 4 * np.pi, n_terms)
    r = series

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, marker='o', markersize=3, linewidth=1.5, label=title)

    for i in range(0, n_terms, num_annotate):
        ax.text(x[i], y[i], f"{label_prefix}{i+1}", fontsize=8, color='blue')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    
    return fig

# Harmonic series: a_n = 1/n
st.header("📙 Harmonic Series Spiral")
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
**Series:** $a_n = \frac{1}{n}$  
**Description:** This plot represents each term of the harmonic series as a point on a spiral path.  
- The angle increases with each term (like winding around a clock).
- The distance from the center reflects the value of each term.
- As $n$ increases, $a_n$ shrinks, and the spiral tightens.
""")

n_terms = st.slider("Number of terms", min_value=10, max_value=300, value=100)
harmonic_series = 1 / np.arange(1, n_terms + 1)

fig = plot_spiral_series(harmonic_series, title="Harmonic Series Spiral", label_prefix="a")
# Create three columns and display the figure in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
with col2:
    st.pyplot(fig)
