import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Geometric Series Visualizer", layout="wide")

st.title("📐 Geometric Series Visualizer")
st.markdown("""
The **Geometric Series** is a series of the form:

""")
st.latex(r"\sum_{n=0}^{\infty} ar^n = a + ar + ar^2 + ar^3 + \dots")

st.markdown("""
Where:
- \( a \) is the first term  
- \( r \) is the common ratio  

The series **converges** when \( |r| < 1 \), and the sum becomes:
""")

st.latex(r"\sum_{n=0}^{\infty} ar^n = \frac{a}{1 - r} \quad \text{if } |r| < 1")

# User inputs
a = st.number_input("First term (a)", value=1.0)
r = st.number_input("Common ratio (r)", value=0.5)
n_terms = st.slider("Number of terms to plot", 1, 100, 20)

# Compute terms and partial sums
terms = np.array([a * r**n for n in range(n_terms)])
partial_sums = np.cumsum(terms)
n = np.arange(n_terms)

# True sum (if converges)
converges = abs(r) < 1
true_sum = a / (1 - r) if converges else None

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=n, y=partial_sums, mode='lines+markers', name='Partial Sum', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=n, y=terms, mode='markers', name='Individual Terms', marker=dict(color='orange', symbol='circle')))

if converges:
    fig.add_trace(go.Scatter(x=n, y=[true_sum]*n_terms, mode='lines', name=f'Limit: {true_sum:.2f}', line=dict(color='green', dash='dash')))

fig.update_layout(title="Geometric Series: Partial Sums & Convergence",
                  xaxis_title="Term Index (n)", yaxis_title="Sum",
                  legend=dict(x=0, y=1))

st.plotly_chart(fig, use_container_width=True)

# Explanation
with st.expander("📘 Real-world Uses of Geometric Series"):
    st.markdown("""
- **Finance**: Calculating compound interest or loan payments.
- **Physics**: Modeling infinite reflections or damping effects.
- **Computer Science**: Analyzing algorithms with repeated operations (e.g., recursive calls).
- **Math**: Building blocks for more advanced series (e.g., power series).
    """)

if not converges:
    st.warning("⚠️ The series does **not converge** because |r| ≥ 1.")

st.markdown("---")
st.header("🔁 3D Visualization of Geometric Series Evolution")

# Inputs
n_terms = st.slider("Number of Terms (n)", min_value=5, max_value=50, value=20)
a = st.slider("Initial Value (a)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
r = st.slider("Common Ratio (r)", min_value=-2.0, max_value=2.0, value=0.5, step=0.05)

# Compute terms and partial sums
n = np.arange(n_terms)
terms = a * r**n
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
        name="Geometric Terms"
    )
])
fig.update_layout(
    title=f"3D Evolution of Geometric Series with a={a}, r={r}",
    scene=dict(
        xaxis_title="Term Index n",
        yaxis_title="Term Value (ar^n)",
        zaxis_title="Partial Sum"
    )
)

st.plotly_chart(fig, use_container_width=True)

# Show theoretical sum if convergent
if abs(r) < 1:
    final_sum = a / (1 - r)
    st.success(f"✅ The series converges to: {final_sum:.4f}")
else:
    st.warning("⚠️ The series diverges since |r| ≥ 1.")

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

# Geometric series: a_n = (1/2)^n
st.header("📘 Geometric Series Spiral")
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
**Series:** $a_n = \left(\frac{1}{2}\right)^n$  
**Description:** This spiral shows the rapid decay of the geometric series.  
- The angle grows uniformly, creating a spiral flow.
- Each new term is half the size of the previous, causing a rapid inward curl.
- The series converges, and visually the spiral pulls quickly toward the origin.
""")

n_terms = st.slider("Number of terms", min_value=10, max_value=300, value=100)
geometric_series = 1 / (2 ** np.arange(1, n_terms + 1))

fig = plot_spiral_series(geometric_series, title="Geometric Series Spiral", label_prefix="a")
# Create three columns and display the figure in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Center column takes 50% width
with col2:
    st.pyplot(fig)
