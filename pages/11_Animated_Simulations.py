import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import tempfile
import math

st.set_page_config(page_title="🎞️ Animated Simulations", layout="wide")
st.title("🎞️ Animated Simulations of Mathematical Series")

st.markdown("""
Choose a series from the dropdown below to see a 3D animated simulation of how its terms evolve in space.
""")

# Dropdown list of series
series_choice = st.selectbox(
    "Select a Mathematical Series to Animate",
    (
        "Taylor Series",
        "Fourier Series",
        "Maclaurin Series",
        "Geometric Series",
        "Harmonic Series",
        "Binomial Series",
        "Power Series",
        "Laurent Series",
        "Legendre Series",
        "Z-Series (Z-Transform)"
    )
)

# Function to generate (x, y, z) data based on selected series
def generate_series_data(series_name):
    n_terms = 50
    x = np.arange(n_terms)
    z = np.linspace(0, 20, n_terms)

    if series_name == "Taylor Series":
        y = np.array([np.sin(1) + ((-1)**n * (1)**(2*n + 1)) / math.factorial(2*n + 1) for n in x])
    elif series_name == "Fourier Series":
        y = np.array([4 / np.pi * sum([np.sin((2*k - 1)*np.pi*n/20)/(2*k - 1) for k in range(1, 10)]) for n in x])
    elif series_name == "Maclaurin Series":
        y = np.array([(1)**n / math.factorial(n) for n in x])  # e^x at x=1
    elif series_name == "Geometric Series":
        y = 2 ** x
    elif series_name == "Harmonic Series":
        y = np.array([np.sum([1 / (k + 1) for k in range(n + 1)]) for n in x])
    elif series_name == "Binomial Series":
        y = np.array([(1 + 0.5) ** n for n in x])  # Binomial expansion of (1 + x)^n at x=0.5
    elif series_name == "Power Series":
        y = np.array([(1 + n * 0.1) ** n for n in x])  # Simplified power series
    elif series_name == "Laurent Series":
        y = np.array([1 / n if n != 0 else 0 for n in x]) + np.array([n for n in x])  # Combo of pos & neg powers
    elif series_name == "Legendre Series":
        y = np.polynomial.legendre.legval(x / n_terms, [1] * 5)
    elif series_name == "Z-Series (Z-Transform)":
        y = np.array([0.9**n for n in x])  # Simple z-transform of a^n

    return x, y, z

# Function to create and return GIF path
def generate_animation(series_name):
    x, y, z = generate_series_data(series_name)
    n_terms = len(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"3D Animation: {series_name}")
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))
    line, = ax.plot([], [], [], lw=2)

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=n_terms, interval=100, blit=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as f:
        ani.save(f.name, writer='pillow')
        return f.name

# Display animation
st.markdown(f"### 📈 {series_choice} 3D Animation")
video_file = generate_animation(series_choice)
st.image(video_file)
