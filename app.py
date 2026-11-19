"""SimilarityX Streamlit application."""

from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from core.semantic_model import SemanticModelError
from utils.similarity import compare_texts


EXAMPLES = [
    ("How do I reset my password?", "I forgot my password; how can I reset it?"),
    ("What is the capital of France?", "Which city is the capital of France?"),
    ("How to bake a chocolate cake?", "The weather is pleasant in summer."),
]


def load_css() -> None:
    """Load the application stylesheet."""
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        try:
            st.markdown(
                f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True,
            )
        except OSError as exc:
            st.warning(f"Custom styling could not be loaded: {exc}")


def initialize_state() -> None:
    """Initialize persistent session state."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "text_a" not in st.session_state:
        st.session_state.text_a = ""
    if "text_b" not in st.session_state:
        st.session_state.text_b = ""
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None


@st.cache_resource(show_spinner=False)
def warm_model_cache():
    """Warm the SentenceTransformer through the backend cache."""
    from core.semantic_model import get_semantic_model

    return get_semantic_model()


def clear_inputs() -> None:
    st.session_state.text_a = ""
    st.session_state.text_b = ""
    st.session_state.latest_result = None


def use_example(index: int) -> None:
    st.session_state.text_a, st.session_state.text_b = EXAMPLES[index]


def render_header() -> None:
    col1, col2 = st.columns([5, 2])
    with col1:
        st.markdown(
            """
            <div class="app-header">
              <div class="brand-mark">SX</div>
              <div>
                <div class="brand-name">SimilarityX</div>
                <div class="brand-subtitle">Text intelligence workspace</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="system-status">
              <span class="status-dot"></span>
              <span>Semantic engine online</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-kicker">WORKSPACE</div>
          <div class="sidebar-title">NLP Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        ["Text Comparison", "Comparison History", "Methodology", "About Project"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
          <div class="footer-label">MODEL</div>
          <div class="footer-value">all-MiniLM-L6-v2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


def result_explanation(classification: str, semantic_score: float, tfidf_score: float) -> str:
    if classification == "Duplicate":
        return (
            f"The texts have highly aligned meaning ({semantic_score:.2f}% semantic similarity). "
            f"The TF-IDF result is {tfidf_score:.2f}%, providing a lexical comparison."
        )
    if classification == "Similar":
        return (
            f"The texts share relevant meaning despite different wording ({semantic_score:.2f}% semantic similarity). "
            f"The TF-IDF overlap is {tfidf_score:.2f}%."
        )
    return (
        f"The texts appear substantially different, with {semantic_score:.2f}% semantic similarity "
        f"and {tfidf_score:.2f}% lexical similarity."
    )


def render_comparison_visualization(semantic_score: float, tfidf_score: float) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[semantic_score, tfidf_score],
            y=["Semantic", "TF-IDF"],
            orientation="h",
            marker=dict(
                color=["#315DFF", "#8B83D6"],
                line=dict(width=0),
            ),
            text=[f"{semantic_score:.2f}%", f"{tfidf_score:.2f}%"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=12, r=40, t=16, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor="#E8E7E1", zeroline=False, tickformat=".0f"),
        yaxis=dict(showgrid=False),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#18233A", size=12),
        bargap=0.58,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_results(result: dict) -> None:
    semantic_score = float(result["semantic_score"])
    tfidf_score = float(result["tfidf_score"])
    classification = str(result["classification"])
    timestamp = result["timestamp"]

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="results-heading">
          <div>
            <div class="eyebrow">COMPARISON RESULT</div>
            <h2>Similarity analysis</h2>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card1, card2, card3 = st.columns([1.25, 1, 1])
    with card1:
        st.markdown(
            f"""
            <div class="metric-card metric-primary">
              <div class="metric-label">Semantic similarity</div>
              <div class="metric-value">{semantic_score:.2f}<span>%</span></div>
              <div class="metric-detail">Sentence embeddings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">TF-IDF similarity</div>
              <div class="metric-value">{tfidf_score:.2f}<span>%</span></div>
              <div class="metric-detail">Lexical comparison</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card3:
        class_name = classification.lower()
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Classification</div>
              <div class="classification-badge {class_name}">{classification}</div>
              <div class="metric-detail">Initial threshold model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    chart_col, explanation_col = st.columns([1.35, 1])
    with chart_col:
        st.markdown('<div class="panel-title">Similarity profile</div>', unsafe_allow_html=True)
        render_comparison_visualization(semantic_score, tfidf_score)
    with explanation_col:
        st.markdown('<div class="panel-title">Interpretation</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="explanation-box">
              <p>{result_explanation(classification, semantic_score, tfidf_score)}</p>
              <div class="timestamp">Analyzed {timestamp}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_text_comparison() -> None:
    st.markdown(
        """
        <div class="page-heading">
          <div class="eyebrow">TEXT ANALYSIS</div>
          <h1>Compare two texts</h1>
          <p>Measure lexical and semantic similarity in a single workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, actions_col = st.columns([5, 1])
    with actions_col:
        st.markdown('<div class="example-label">Try an example</div>', unsafe_allow_html=True)
        for index in range(len(EXAMPLES)):
            st.button(f"Example {index + 1}", key=f"example_{index}", use_container_width=True,
                      on_click=use_example, args=(index,))

    first_col, second_col = st.columns(2)
    with first_col:
        st.markdown('<div class="input-panel-label">FIRST TEXT</div>', unsafe_allow_html=True)
        text_a = st.text_area(
            "First text",
            key="text_a",
            height=190,
            max_chars=2000,
            placeholder="Enter a sentence or question...",
            label_visibility="collapsed",
        )
        st.markdown(f'<div class="character-count">{len(text_a)} / 2000 characters</div>', unsafe_allow_html=True)

    with second_col:
        st.markdown('<div class="input-panel-label">SECOND TEXT</div>', unsafe_allow_html=True)
        text_b = st.text_area(
            "Second text",
            key="text_b",
            height=190,
            max_chars=2000,
            placeholder="Enter another sentence or question...",
            label_visibility="collapsed",
        )
        st.markdown(f'<div class="character-count">{len(text_b)} / 2000 characters</div>', unsafe_allow_html=True)

    compare_col, clear_col, _ = st.columns([1.05, 0.75, 3.2])
    with compare_col:
        compare_clicked = st.button("Compare text", type="primary", use_container_width=True)
    with clear_col:
        st.button("Clear all", use_container_width=True, on_click=clear_inputs)

    if compare_clicked:
        if not text_a.strip() or not text_b.strip():
            st.warning("Enter both texts before starting a comparison.")
        else:
            with st.spinner("Analyzing sentence meaning..."):
                try:
                    warm_model_cache()
                    result = compare_texts(text_a, text_b)
                    result["timestamp"] = datetime.now().strftime("%d %b %Y, %H:%M")
                    st.session_state.latest_result = result
                    st.session_state.history.insert(
                        0,
                        {
                            "text_a": text_a,
                            "text_b": text_b,
                            "tfidf_score": result["tfidf_score"],
                            "semantic_score": result["semantic_score"],
                            "classification": result["classification"],
                            "timestamp": result["timestamp"],
                        },
                    )
                    st.session_state.history = st.session_state.history[:50]
                except SemanticModelError as exc:
                    st.error(str(exc))
                except ValueError as exc:
                    st.error(f"Invalid comparison: {exc}")
                except Exception as exc:
                    st.error(f"Comparison could not be completed: {exc}")

    if st.session_state.latest_result is not None:
        render_results(st.session_state.latest_result)


def render_history() -> None:
    st.markdown(
        """
        <div class="page-heading">
          <div class="eyebrow">SESSION DATA</div>
          <h1>Comparison history</h1>
          <p>Review comparisons made during this session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-title">No comparisons yet</div>
              <div class="empty-text">Completed analyses will appear here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.latest_result = None
        st.rerun()

    for item in st.session_state.history:
        st.markdown(
            f"""
            <div class="history-card">
              <div class="history-top">
                <div class="classification-badge {item["classification"].lower()}">{item["classification"]}</div>
                <div class="history-time">{item["timestamp"]}</div>
              </div>
              <div class="history-sentences">
                <div><span>01</span>{item["text_a"]}</div>
                <div><span>02</span>{item["text_b"]}</div>
              </div>
              <div class="history-scores">
                <div><strong>{item["semantic_score"]:.2f}%</strong><small>Semantic</small></div>
                <div><strong>{item["tfidf_score"]:.2f}%</strong><small>TF-IDF</small></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_methodology() -> None:
    st.markdown(
        """
        <div class="page-heading">
          <div class="eyebrow">REFERENCE</div>
          <h1>Methodology</h1>
          <p>How SimilarityX measures textual relationship.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="info-card">
              <div class="info-number">01</div>
              <h3>TF-IDF</h3>
              <p>Term Frequency-Inverse Document Frequency represents words by how
              distinctive they are within the compared text pair.</p>
            </div>
            <div class="info-card">
              <div class="info-number">02</div>
              <h3>Cosine similarity</h3>
              <p>Cosine similarity measures the angle between vector representations,
              producing scores from zero to complete overlap for TF-IDF and embeddings.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="info-card">
              <div class="info-number">03</div>
              <h3>Sentence Transformers</h3>
              <p>The application uses all-MiniLM-L6-v2 to transform each normalized
              sentence into a dense semantic embedding.</p>
            </div>
            <div class="info-card">
              <div class="info-number">04</div>
              <h3>Classification</h3>
              <p>Semantic scores at 85% or above are Duplicate, scores from 60% to
              below 85% are Similar, and lower scores are Different. These are initial thresholds.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about() -> None:
    st.markdown(
        """
        <div class="page-heading">
          <div class="eyebrow">PROJECT</div>
          <h1>About SimilarityX</h1>
          <p>A focused application for identifying lexical and semantic text overlap.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="about-grid">
          <section>
            <h3>Objective</h3>
            <p>SimilarityX compares two sentences or questions and determines whether
            they are duplicates, similar, or different using traditional and semantic NLP approaches.</p>
          </section>
          <section>
            <h3>Technology</h3>
            <p>Python · Streamlit · Pandas · NumPy · Scikit-learn · NLTK ·
            Sentence Transformers · Plotly · Custom CSS</p>
          </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="SimilarityX",
        page_icon="SX",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()
    initialize_state()
    render_header()
    page = render_sidebar()

    if page == "Text Comparison":
        render_text_comparison()
    elif page == "Comparison History":
        render_history()
    elif page == "Methodology":
        render_methodology()
    else:
        render_about()


if __name__ == "__main__":
    main()
