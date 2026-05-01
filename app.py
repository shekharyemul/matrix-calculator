import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def check_definiteness(eigenvalues, is_symmetric):
    if not is_symmetric:
        return "N/A (Matrix not symmetric)", "⚪"
    real_parts = np.real(eigenvalues)
    if np.all(real_parts > 1e-10):
        return "Positive Definite", "🟢"
    elif np.all(real_parts < -1e-10):
        return "Negative Definite", "🔴"
    elif np.all(real_parts >= -1e-10) and np.any(abs(real_parts) < 1e-10):
        return "Positive Semi-Definite", "🎾"
    elif np.all(real_parts <= 1e-10) and np.any(abs(real_parts) < 1e-10):
        return "Negative Semi-Definite", "🟤"
    else:
        return "Indefinite", "🟡"

def matrix_to_latex(arr):
    """Converts a numpy array to a LaTeX bmatrix."""
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
        
    lines = []
    for row in arr:
        row_str = []
        for val in row:
            if np.iscomplex(val):
                # If imaginary part is very small, treat as real
                if abs(np.imag(val)) < 1e-10:
                    row_str.append(f"{np.real(val):.4g}")
                else:
                    row_str.append(f"{np.real(val):.4g}{np.imag(val):+.4g}i")
            else:
                row_str.append(f"{np.real(val):.4g}")
        lines.append(" & ".join(row_str))
        
    return r"\begin{bmatrix} " + r" \\ ".join(lines) + r" \end{bmatrix}"

st.set_page_config(
    page_title="Matrix Diagonalization & Quadratic Form", 
    page_icon="🧮", 
    layout="wide"
)

# Custom CSS for a professional and attractive look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d2ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.6);
        color: white;
    }
    
    /* Data editor container */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #333;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧮 Matrix Diagonalization & Quadratic Form Calculator")
st.markdown("Analyze square matrices by transforming them into their diagonal form and computing their associated quadratic equations. This is highly useful for understanding the principal axes of a quadratic form.")

st.sidebar.header("⚙️ Matrix Configuration")
n = st.sidebar.number_input("Matrix Dimension (n x n)", min_value=2, max_value=6, value=3, step=1)

st.subheader(f"Input the {n}x{n} Matrix Elements")
st.markdown("Fill in the values below. For a valid standard quadratic form, the matrix should typically be **symmetric** ($A = A^T$).")

# Initialize default symmetric matrix (identity matrix)
if 'n' not in st.session_state or st.session_state.n != n:
    st.session_state.n = n
    st.session_state.default_matrix = np.eye(n)

# Create an editable dataframe for matrix input
df = pd.DataFrame(st.session_state.default_matrix, columns=[f"Col {i+1}" for i in range(n)])

# Display data editor
edited_df = st.data_editor(
    df, 
    use_container_width=True, 
    num_rows="fixed",
    hide_index=True
)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Calculate Diagonalization"):
    with st.spinner("Calculating Eigenvalues and Eigenvectors..."):
        try:
            # Convert input to numpy array
            matrix = edited_df.to_numpy().astype(float)
            
            # Check if the matrix is symmetric
            is_symmetric = np.allclose(matrix, matrix.T)
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            
            # For symmetric matrices, eigenvectors are orthogonal, but let's check if diagonalizable generally
            # If we have n linearly independent eigenvectors, it's diagonalizable
            cond_number = np.linalg.cond(eigenvectors)
            if cond_number > 1e10:
                st.warning("⚠️ This matrix might not be diagonalizable (it may be defective). The results below might be numerically unstable.")
            
            # Create diagonal matrix D
            D = np.diag(eigenvalues)
            
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["📊 Spectral Decomposition", "📐 Orthogonality & Definiteness", "🌌 3D Visualizer"])
            
            with tab1:
                st.markdown("### ➗ Matrix Decomposition ($A = P D P^{-1}$)")
                st.info("The original matrix $A$ is decomposed into a diagonal matrix $D$ and an eigenvector matrix $P$.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Original Matrix ($A$)")
                    st.latex(r"A = " + matrix_to_latex(matrix))
                    st.markdown("#### Eigenvalues ($\lambda$)")
                    st.latex(r"\lambda = " + matrix_to_latex(eigenvalues))
                with col2:
                    st.markdown("#### Diagonalized Matrix ($D$)")
                    st.latex(r"D = " + matrix_to_latex(D))
                    st.markdown("#### Eigenvector Matrix ($P$)")
                    st.latex(r"P = " + matrix_to_latex(eigenvectors))
                    
                st.markdown("---")
                st.markdown("### 🔢 Quadratic Form Transformation")
                st.markdown("The original quadratic form $Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ transforms along the principal axes to $Q(\mathbf{y}) = \mathbf{y}^T D \mathbf{y}$:")
                
                # Construct the quadratic equation string
                terms = []
                for i, val in enumerate(eigenvalues):
                    if np.iscomplex(val):
                        if abs(np.imag(val)) < 1e-10:
                            val_real = np.real(val)
                            if abs(val_real) < 1e-10: continue
                            val_str = f"{val_real:.4g}"
                        else:
                            val_str = f"({np.real(val):.4g}{np.imag(val):+.4g}i)"
                    else:
                        val_real = np.real(val)
                        if abs(val_real) < 1e-10: continue
                        val_str = f"{val_real:.4g}"
                        
                    terms.append(f"{val_str} y_{{{i+1}}}^2")
                    
                equation = " + ".join(terms).replace("+ -", "- ") if terms else "0"
                st.latex(r"Q(\mathbf{y}) = " + equation)

            with tab2:
                st.markdown("### 🔍 Definiteness Analysis")
                def_status, def_icon = check_definiteness(eigenvalues, is_symmetric)
                st.metric("Matrix Classification", f"{def_icon} {def_status}")
                
                if is_symmetric:
                    st.markdown("---")
                    st.markdown("### 📐 The Spectral Theorem Proof")
                    st.success("**Spectral Theorem:** Every real symmetric matrix is orthogonally diagonalizable.")
                    st.markdown("Because $A$ is symmetric, its eigenvectors form an orthogonal basis. This means the matrix $P$ is an **orthogonal matrix**, so $P^{-1} = P^T$ and therefore $P^T P = I$. Let's dynamically verify this!")
                    
                    # Calculate P^T P
                    PtP = eigenvectors.T @ eigenvectors
                    # Clean up small floating point errors for display
                    PtP[np.abs(PtP) < 1e-10] = 0
                    
                    st.markdown("#### Calculating $P^T P$:")
                    st.latex(r"P^T P \approx " + matrix_to_latex(PtP))
                    
                    if np.allclose(PtP, np.eye(n), atol=1e-5):
                        st.info("✅ **Proof Complete:** As shown above, $P^T P$ is indeed the Identity Matrix. The eigenvectors are orthonormal!")
                else:
                    st.warning("📝 **Note:** The entered matrix is not symmetric. Standard orthogonal diagonalization (where $P^T P = I$) requires a symmetric matrix.")

            with tab3:
                # --- Crazy Surprise Feature: 3D Visualizations ---
                if n in [2, 3] and is_symmetric:
                    st.markdown("### 🌌 Interactive 3D Visualizer")
                    st.markdown("This interactive plot reveals how the matrix transforms a unit sphere (for 3D) or unit circle (for 2D) into an ellipsoid. This elegantly proves how the **eigenvectors** geometrically act as the **principal axes** of the transformed shape.")
                    
                    fig = go.Figure()
                    
                    if n == 2:
                        # Plot 3D surface z = x^T A x
                        x_vals = np.linspace(-2, 2, 50)
                        y_vals = np.linspace(-2, 2, 50)
                        X, Y = np.meshgrid(x_vals, y_vals)
                        Z = matrix[0,0]*X**2 + (matrix[0,1]+matrix[1,0])*X*Y + matrix[1,1]*Y**2
                        
                        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, showscale=False))
                        
                        # Add eigenvectors on the z=0 plane
                        for i in range(2):
                            ev = eigenvectors[:, i] * max(1, np.abs(np.real(eigenvalues[i])))
                            fig.add_trace(go.Scatter3d(
                                x=[0, np.real(ev[0])], y=[0, np.real(ev[1])], z=[0, 0],
                                mode='lines+text',
                                line=dict(width=10, color=['red', 'yellow'][i]),
                                text=['', f'v{i+1}'],
                                name=f"Eigenvector {i+1} (λ={np.real(eigenvalues[i]):.2f})"
                            ))
                            
                        fig.update_layout(
                            title="Quadratic Surface $z = \mathbf{x}^T A \mathbf{x}$", 
                            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                            height=600,
                            margin=dict(l=0, r=0, b=0, t=40)
                        )
                        
                    elif n == 3:
                        # 3D case, plot transformed unit sphere (Ellipsoid)
                        u = np.linspace(0, 2 * np.pi, 50)
                        v = np.linspace(0, np.pi, 50)
                        x = np.outer(np.cos(u), np.sin(v))
                        y = np.outer(np.sin(u), np.sin(v))
                        z = np.outer(np.ones(np.size(u)), np.cos(v))
                        
                        points = np.vstack((x.flatten(), y.flatten(), z.flatten()))
                        transformed_points = matrix @ points
                        
                        X_t = transformed_points[0, :].reshape(50, 50)
                        Y_t = transformed_points[1, :].reshape(50, 50)
                        Z_t = transformed_points[2, :].reshape(50, 50)
                        
                        fig.add_trace(go.Surface(x=X_t, y=Y_t, z=Z_t, colorscale='Plasma', opacity=0.6, showscale=False))
                        
                        # Add eigenvectors as principal axes
                        max_val = np.max(np.abs(eigenvalues))
                        if max_val < 1e-10: max_val = 1
                        
                        for i in range(3):
                            ev = eigenvectors[:, i] * np.real(eigenvalues[i])
                            if np.abs(eigenvalues[i]) < 1e-10:
                                ev = eigenvectors[:, i] * max_val * 0.1
                                
                            fig.add_trace(go.Scatter3d(
                                x=[0, np.real(ev[0])], 
                                y=[0, np.real(ev[1])], 
                                z=[0, np.real(ev[2])],
                                mode='lines',
                                line=dict(width=10, color=['red', 'green', 'cyan'][i]),
                                name=f"Principal Axis {i+1} (λ={np.real(eigenvalues[i]):.2f})"
                            ))
                            
                        fig.update_layout(
                            title="Transformed Unit Sphere (Ellipsoid) with Principal Axes", 
                            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode='data'), 
                            height=700, 
                            margin=dict(l=0, r=0, b=0, t=40)
                        )
                        
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("✨ *Notice how the eigenvectors perfectly align with the geometric principal axes of the transformed shape!*")
                else:
                    st.info("⚠️ The Interactive 3D Visualizer requires the matrix to be a **symmetric $2 \\times 2$ or $3 \\times 3$ matrix**.")
                
        except ValueError:
            st.error("❌ Invalid input detected. Please ensure all matrix elements are numeric.")
        except np.linalg.LinAlgError:
            st.error("❌ Matrix computation failed. It might not be diagonalizable.")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
