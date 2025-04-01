import streamlit as st

st.set_page_config(page_title="📚 Mathematical Series Hub", layout="wide")

st.title("📚 Welcome to the Mathematical Series Visualizer")
st.markdown("""
Explore the beauty and power of mathematical series through interactive visualizations.  
Each listed series includes:
- 🧠 A plain-language explanation  
- 🔣 Its mathematical definition (with LaTeX!)  
- 🌍 Where it shows up in the real world  
- 🎯 A link to an interactive illustration
""")

st.page_link("pages/11_Animated_Simulations.py", label="🎞️ Go to Animated Simulations Page", icon="🎬")

# Utility function for a series block
def series_block(number, name, formula, explanation, applications, page_path):
    st.markdown(f"### {number}. {name}")
    st.latex(formula)
    st.markdown(explanation)
    st.markdown("**Applications:**")
    st.markdown(applications)
    st.page_link(page_path, label=f"👉 Go to {name} Visualizer", icon="🔗")
    st.divider()


# 1. Taylor Series
series_block(
    1,
    "Taylor Series",
    r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x - a)^n",
    "Taylor Series approximates smooth functions using their derivatives at a point. It's widely used in physics, calculators, and simulations.",
    "- Numerical computation and calculators\n- Physics simulations\n- Engineering approximations",
    "pages/1_Taylor_Series.py"
)

# 2. Fourier Series
series_block(
    2,
    "Fourier Series",
    r"f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[a_n \cos(nx) + b_n \sin(nx)\right]",
    "Fourier Series breaks any periodic signal into sine and cosine components.",
    "- Sound and signal processing\n- Image compression\n- Solving PDEs",
    "pages/2_Fourier_Series.py"
)

# 3. Maclaurin Series
series_block(
    3,
    "Maclaurin Series",
    r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!} x^n",
    "Special case of Taylor Series centered at 0. Common for functions like sin(x), cos(x), and e^x.",
    "- Scientific calculators\n- Series expansions in ML\n- Computer algebra systems",
    "pages/3_Maclaurin_Series.py"
)

# 4. Geometric Series
series_block(
    4,
    "Geometric Series",
    r"\sum_{n=0}^{\infty} ar^n = \frac{a}{1 - r} \quad \text{if } |r| < 1",
    "One of the simplest and most fundamental infinite series. It converges when |r| < 1.",
    "- Compound interest\n- Recursive processes\n- Physics (damping, optics)",
    "pages/4_Geometric_Series.py"
)

# 5. Harmonic Series
series_block(
    5,
    "Harmonic Series",
    r"\sum_{n=1}^{\infty} \frac{1}{n}",
    "Though the terms approach zero, the harmonic series diverges very slowly.",
    "- Algorithm analysis (e.g. Quicksort)\n- Signal delay models\n- Mathematical divergence examples",
    "pages/5_Harmonic_Series.py"
)

# 6. Binomial Series
series_block(
    6,
    "Binomial Series",
    r"(1 + x)^k = \sum_{n=0}^{\infty} \binom{k}{n} x^n",
    "Expands binomials with fractional or negative powers using infinite series.",
    "- Finance (option pricing)\n- Physics approximations\n- Error bounds in computations",
    "pages/6_Binomial_Series.py"
)

# 7. Power Series
series_block(
    7,
    "Power Series",
    r"\sum_{n=0}^{\infty} c_n (x - a)^n",
    "General form of a function written as an infinite polynomial. Framework for Taylor, Maclaurin, Binomial.",
    "- Function approximation\n- ML (series-based models)\n- Simulation of analytic functions",
    "pages/7_Power_Series.py"
)

# 8. Laurent Series
series_block(
    8,
    "Laurent Series",
    r"f(x) = \sum_{n=-\infty}^{\infty} c_n (x - a)^n",
    "Like Taylor, but allows negative powers — useful near singularities in complex analysis.",
    "- Residue calculus\n- Electrical engineering\n- Complex dynamics",
    "pages/8_Laurent_Series.py"
)

# 9. Legendre Series
series_block(
    9,
    "Legendre Series",
    r"f(x) = \sum_{n=0}^{\infty} a_n P_n(x)",
    "Represents a function using Legendre polynomials — orthogonal on [-1, 1].",
    "- Quantum mechanics\n- Spherical physics problems\n- Signal modeling",
    "pages/9_Legendre_Series.py"
)

# 10. Z-Series (Z-Transform)
series_block(
    10,
    "Z-Series (Z-Transform)",
    r"X(z) = \sum_{n=0}^{\infty} x[n] z^{-n}",
    "Transforms discrete signals into complex frequency domain — key in digital signal processing.",
    "- DSP and filtering\n- Control systems\n- Communications encoding",
    "pages/10_Z_Series.py"
)

# Footer
st.markdown("---")
st.markdown("🚀 Built by **Shubham Kandpal** — [LinkedIn](https://linkedin.com/in/shubham-kandpal) | [GitHub](https://github.com/Shubham111Kandpal)")
