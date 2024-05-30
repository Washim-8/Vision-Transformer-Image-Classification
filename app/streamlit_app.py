# =============================================================================
# app/streamlit_app.py
# Premium Streamlit web interface for the Vision Transformer classifier.
#
# Features:
#   • Upload any image → get real-time CIFAR-10 predictions
#   • Confidence bar chart
#   • Model architecture summary
#   • Training metrics viewer (if curves are available)
#   • Attention-map placeholder for future extension
#
# Run:
#   streamlit run app/streamlit_app.py
# =============================================================================

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import CLASS_NAMES, MODEL_PATH, GRAPHS_DIR, DEVICE
from app.predict_image import predict, load_model, INFERENCE_TRANSFORM

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Transformer – CIFAR-10 Classifier",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for premium look ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Dark gradient background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(12px);
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.4);
    }

    /* ── Hero header ── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* ── Prediction card ── */
    .pred-card {
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(16px);
    }
    .pred-label {
        font-size: 2rem;
        font-weight: 700;
        color: #818cf8;
    }
    .pred-conf {
        font-size: 1.2rem;
        color: #c084fc;
    }

    /* ── Info badge ── */
    .badge {
        display: inline-block;
        background: rgba(16,185,129,0.15);
        border: 1px solid rgba(16,185,129,0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #34d399;
        margin: 4px;
    }

    /* ── Section header ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 4px solid #6366f1;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Confidence bar ── */
    .conf-row { margin: 4px 0; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    topk_slider = st.slider("Top-K predictions", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### 📋 Model Info")
    st.markdown("""
    <span class="badge">ViT Architecture</span>
    <span class="badge">CIFAR-10</span>
    <span class="badge">PyTorch</span>
    """, unsafe_allow_html=True)
    st.markdown("""
    **Patch size:** 16 × 16  
    **Image size:** 224 × 224  
    **Classes:** 10  
    """)

    st.markdown("---")
    st.markdown("### 🏷️ CIFAR-10 Classes")
    class_icons = ["✈️","🚗","🐦","🐱","🦌","🐶","🐸","🐴","🚢","🚚"]
    for icon, name in zip(class_icons, CLASS_NAMES):
        st.markdown(f"{icon} **{name.capitalize()}**")

    st.markdown("---")
    device_icon = "⚡" if str(DEVICE) == "cuda" else "💻"
    st.markdown(f"{device_icon} Running on **{str(DEVICE).upper()}**")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔭 Vision Transformer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Image Classification with ViT · Trained on CIFAR-10 · Powered by PyTorch</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_arch, tab_about = st.tabs([
    "🖼️  Predict", "📈  Training Metrics", "🏗️  Architecture", "👨‍💻  About & Contact"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_upload, col_result = st.columns([1, 1.4], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Input Image", use_container_width=True)
            predict_btn = st.button("🚀  Classify Image", type="primary")
        else:
            st.info("👆 Upload an image to get started", icon="ℹ️")
            predict_btn = False

    with col_result:
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

        if uploaded_file and predict_btn:
            # Check if model exists
            if not os.path.exists(MODEL_PATH):
                st.error(
                    "⚠️ No trained model found!  "
                    "Run `python main.py` to train the model first.",
                    icon="🚨"
                )
            else:
                with st.spinner("Running inference…"):
                    # Save temp file so predict() can open it with PIL
                    tmp_path = "/tmp/vit_input_image.jpg"
                    image.save(tmp_path)
                    results  = predict(tmp_path, topk=topk_slider)

                top = results[0]
                class_icons_map = {
                    c: i for c, i in zip(CLASS_NAMES, ["✈️","🚗","🐦","🐱","🦌","🐶","🐸","🐴","🚢","🚚"])
                }

                # ── Top prediction card ────────────────────────────────────
                st.markdown(f"""
                <div class="pred-card">
                    <div style="font-size:3rem">{class_icons_map.get(top['class'], '🎯')}</div>
                    <div class="pred-label">{top['class'].upper()}</div>
                    <div class="pred-conf">{top['confidence']:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                # ── Confidence bar chart ───────────────────────────────────
                st.markdown("**Top-K Confidence Scores**")

                labels  = [r["class"].capitalize() for r in results]
                confs   = [r["confidence"] for r in results]
                colors  = ["#6366f1" if i == 0 else "#334155" for i in range(len(results))]

                fig, ax = plt.subplots(figsize=(6, max(2, len(results) * 0.55)))
                fig.patch.set_facecolor("none")
                ax.set_facecolor("none")

                bars = ax.barh(labels[::-1], confs[::-1], color=colors[::-1],
                               edgecolor="none", height=0.6)
                for bar, conf in zip(bars, confs[::-1]):
                    ax.text(
                        min(conf + 1, 99), bar.get_y() + bar.get_height() / 2,
                        f"{conf:.1f}%", va="center", ha="left",
                        fontsize=9, color="#e2e8f0"
                    )

                ax.set_xlim(0, 105)
                ax.tick_params(colors="#94a3b8", labelsize=9)
                ax.spines[:].set_visible(False)
                ax.set_xlabel("Confidence (%)", color="#94a3b8", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, transparent=True)
                plt.close()

        elif not uploaded_file:
            # Placeholder
            st.markdown("""
            <div style="height:280px; display:flex; align-items:center;
                        justify-content:center; border:2px dashed rgba(255,255,255,0.1);
                        border-radius:16px; color:#475569; font-size:1.1rem;">
                Predictions will appear here
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – TRAINING METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown('<div class="section-header">Training History</div>', unsafe_allow_html=True)

    curves_path = os.path.join(GRAPHS_DIR, "training_curves.png")
    conf_path   = os.path.join(GRAPHS_DIR, "confusion_matrix.png")
    preds_path  = os.path.join(GRAPHS_DIR, "sample_predictions.png")

    if os.path.exists(curves_path):
        st.image(curves_path, caption="Loss & Accuracy Curves", use_container_width=True)
    else:
        st.info("Training curves will appear here after you run the training pipeline.", icon="📊")

    col_conf, col_samp = st.columns(2, gap="medium")
    with col_conf:
        if os.path.exists(conf_path):
            st.image(conf_path, caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix will appear after evaluation.", icon="🗃️")
    with col_samp:
        if os.path.exists(preds_path):
            st.image(preds_path, caption="Sample Predictions", use_container_width=True)
        else:
            st.info("Sample predictions will appear after evaluation.", icon="🖼️")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown('<div class="section-header">Vision Transformer Architecture</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        #### 🔷 How ViT Works
        1. **Patch Embedding** – The input image (224×224) is divided into 196 non-overlapping 16×16 patches.  
           Each patch is flattened and projected to a 768-dimensional embedding via a learned convolution.

        2. **[CLS] Token** – A learnable class token is prepended to the patch sequence.  
           This token aggregates global information and is used for classification.

        3. **Positional Encoding** – Learnable position embeddings are added to each patch embedding  
           so the model knows *where* each patch comes from.

        4. **Transformer Encoder** – A stack of 12 encoder blocks, each composed of:
           - Multi-Head Self-Attention (12 heads)
           - Feed-Forward Network (GELU activation)
           - Residual connections & Layer Normalization (pre-norm variant)

        5. **Classification Head** – The [CLS] token's output is passed through an MLP  
           with GELU activation to produce class logits.  Softmax gives probabilities.
        """)

    with col_b:
        st.markdown("#### 📊 Model Variants")
        import pandas as pd
        variants_df = pd.DataFrame({
            "Variant":   ["ViT-Tiny",  "ViT-Small", "ViT-Base"],
            "Embed Dim": [192,          384,          768],
            "Heads":     [3,            6,            12],
            "Layers":    [12,           12,           12],
            "MLP Dim":   [768,          1536,         3072],
            "Params":    ["~5.7M",      "~22M",       "~86M"],
        })
        st.dataframe(variants_df, use_container_width=True, hide_index=True)

        st.markdown("#### 🔧 Training Config")
        st.code("""
Optimizer  : AdamW
Loss       : CrossEntropyLoss (label_smoothing=0.1)
LR         : 3e-4 (Warm-up + Cosine Annealing)
Batch Size : 64
Epochs     : 20 (early stopping patience=5)
Augment    : RandomCrop, HorizontalFlip, ColorJitter
        """, language="yaml")

    st.markdown("---")
    st.markdown("""
    > **Self-Attention** allows every patch to attend to every other patch,  
    > capturing global relationships that traditional CNNs struggle with on small images.
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – ABOUT & CONTACT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="section-header">👨‍💻 About Me</div>', unsafe_allow_html=True)
    
    col_about, col_contact = st.columns([1, 1], gap="large")
    
    with col_about:
        # Profile Card
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 28px;
            backdrop-filter: blur(12px);
        ">
            <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 24px;">
                <div style="
                    width: 72px;
                    height: 72px;
                    background: linear-gradient(135deg, #6366f1, #8b5cf6);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 36px;
                ">👨‍💻</div>
                <div>
                    <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700; color: #e2e8f0;">Washim Shaikh</h2>
                    <p style="margin: 4px 0 0 0; color: #818cf8; font-weight: 500;">Aspiring Software Engineer</p>
                </div>
            </div>
            <p style="color: #94a3b8; line-height: 1.7; margin: 0;">
                Passionate about <strong style="color: #e2e8f0;">AI, Machine Learning, and Web Development</strong>. 
                I have built <strong style="color: #e2e8f0;">18+ real-world projects</strong> including ML systems, 
                full-stack web applications, and IoT-based solutions. My focus is on solving real-world problems 
                using intelligent systems.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Skills Section
        st.subheader("🛠️ Technical Skills")
        
        skills_data = [
            ("💻", "Languages", "Python, Java, C, C++"),
            ("🤖", "ML/AI", "PyTorch, TensorFlow, Scikit-learn"),
            ("🌐", "Web", "HTML, CSS, JavaScript, PHP"),
            ("🗄️", "Database", "MySQL"),
            ("⚙️", "Tools", "Git, GitHub"),
        ]
        
        for icon, category, skills in skills_data:
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 14px 18px;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                margin-bottom: 10px;
            ">
                <span style="font-size: 1.4rem;">{icon}</span>
                <div>
                    <span style="font-weight: 600; color: #e2e8f0; display: block; font-size: 0.9rem;">{category}</span>
                    <span style="color: #94a3b8; font-size: 0.85rem;">{skills}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_contact:
        st.subheader("📬 Contact Me")
        
        # Contact Card
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 28px;
            backdrop-filter: blur(12px);
            text-align: center;
        ">
            <h3 style="margin: 0 0 8px 0; color: #e2e8f0;">Let's Connect!</h3>
            <p style="color: #94a3b8; margin: 0 0 24px 0;">Open to internships, freelance work, and collaborations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Contact Methods
        contact_methods = [
            ("📧", "Email", "washimshaikh33@gmail.com", "mailto:washimshaikh33@gmail.com"),
            ("📱", "Phone", "+91 8884958185", "tel:+918884958185"),
        ]
        
        for icon, label, value, link in contact_methods:
            st.markdown(f"""
            <a href="{link}" style="text-decoration: none;">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 14px;
                    padding: 16px 20px;
                    background: rgba(255,255,255,0.04);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 12px;
                    margin-bottom: 12px;
                    transition: all 0.3s ease;
                " onmouseover="this.style.background='rgba(99,102,241,0.1)'" onmouseout="this.style.background='rgba(255,255,255,0.04)'">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <div>
                        <span style="font-weight: 600; color: #e2e8f0; display: block; font-size: 0.85rem;">{label}</span>
                        <span style="color: #94a3b8; font-size: 0.95rem;">{value}</span>
                    </div>
                </div>
            </a>
            """, unsafe_allow_html=True)
        
        # Social Links
        st.markdown("""
        <div style="display: flex; gap: 12px; margin-top: 20px;">
            <a href="https://github.com/Washim-8" target="_blank" style="flex: 1; text-decoration: none;">
                <button style="
                    width: 100%;
                    padding: 14px 20px;
                    background: linear-gradient(135deg, #374151, #4B5563);
                    border: none;
                    border-radius: 10px;
                    color: white;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                    💻 GitHub
                </button>
            </a>
            <a href="https://www.linkedin.com/in/washim-shaikh-349868281/" target="_blank" style="flex: 1; text-decoration: none;">
                <button style="
                    width: 100%;
                    padding: 14px 20px;
                    background: linear-gradient(135deg, #0077B5, #0A66C2);
                    border: none;
                    border-radius: 10px;
                    color: white;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                    💼 LinkedIn
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Star message
        st.markdown("""
        <div style="
            text-align: center;
            padding: 20px;
            background: rgba(99,102,241,0.08);
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 12px;
        ">
            <p style="margin: 0; color: #818cf8; font-size: 0.95rem;">
                ⭐ Support this project by giving it a star on GitHub!
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ using PyTorch · Vision Transformer (ViT) · Streamlit<br>
    CIFAR-10 Image Classification  |  ML Internship Portfolio Project
</div>
""", unsafe_allow_html=True)
