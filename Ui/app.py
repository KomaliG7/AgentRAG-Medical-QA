# ui/app.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from pipeline.agentrag import run_agentrag

# ── Colour map for confidence labels ────────────────────────
LABEL_COLOURS = {
    "✅ VERIFIED" : "#1a7a1a",
    "🟡 PARTIAL"  : "#b8860b",
    "🔵 INFERRED" : "#1a5fa8",
    "❌ UNKNOWN"  : "#cc2200",
}

def process_query(user_query: str):
    """Runs AgentRAG and returns formatted HTML components."""

    if not user_query.strip():
        return (
            "<p style='color:grey'>Please enter a question.</p>",
            "", "", "", "", ""
        )

    try:
        result = run_agentrag(user_query, verbose=False)
    except Exception as e:
        return (
            f"<p style='color:red'>Error: {str(e)}</p>",
            "", "", "", "", ""
        )

    # ── Confidence Badge ─────────────────────────────────────
    label  = result["label"]
    colour = LABEL_COLOURS.get(label, "#555")
    confidence_html = f"""
    <div style='
        background:{colour};
        color:white;
        padding:10px 20px;
        border-radius:8px;
        font-size:18px;
        font-weight:bold;
        display:inline-block;
        margin-bottom:8px;
    '>{label}</div>
    <p style='color:#555;margin:4px 0 0 2px;font-size:14px'>{result["reason"]}</p>
    <p style='color:#888;font-size:13px'>Retrieval Confidence Score: {result["avg_confidence"]}</p>
    """

    # ── Answer ───────────────────────────────────────────────
    answer_html = f"""
    <div style='
        background:#f9f9f9;
        border-left:4px solid {colour};
        padding:16px;
        border-radius:6px;
        font-size:15px;
        line-height:1.7;
        white-space:pre-wrap;
    '>{result["answer"]}</div>
    """

    # ── Sub-queries ──────────────────────────────────────────
    sq_items = "".join([
        f"<li style='margin:6px 0;color:#333'>{q}</li>"
        for q in result["sub_queries"]
    ])
    subquery_html = f"""
    <div style='background:#eef4ff;padding:12px;border-radius:6px'>
        <b style='color:#1a5fa8'>Query decomposed into {len(result["sub_queries"])} sub-queries:</b>
        <ol style='margin:8px 0 0 0;padding-left:20px'>{sq_items}</ol>
    </div>
    """

    # ── Sources ──────────────────────────────────────────────
    if result["sources"]:
        src_rows = "".join([
            f"""
            <tr style='border-bottom:1px solid #eee'>
                <td style='padding:8px;color:#1a5fa8;font-weight:bold'>[{i+1}]</td>
                <td style='padding:8px'>PubMed {s['pubmed_id']}</td>
                <td style='padding:8px'>{s['year']}</td>
                <td style='padding:8px'>{s['score']}</td>
                <td style='padding:8px;font-size:13px;color:#555'>{s['text'][:120]}...</td>
            </tr>
            """
            for i, s in enumerate(result["sources"][:5])
        ])
        sources_html = f"""
        <table style='width:100%;border-collapse:collapse;font-size:14px'>
            <tr style='background:#f0f0f0;font-weight:bold'>
                <td style='padding:8px'>#</td>
                <td style='padding:8px'>PubMed ID</td>
                <td style='padding:8px'>Year</td>
                <td style='padding:8px'>Score</td>
                <td style='padding:8px'>Passage</td>
            </tr>
            {src_rows}
        </table>
        """
    else:
        sources_html = "<p style='color:grey'>No sources retrieved.</p>"

    # ── Contradictions ───────────────────────────────────────
    if result["contradictions"]:
        c_items = "".join([
            f"<li style='margin:6px 0;color:#8B0000'>{c}</li>"
            for c in result["contradictions"]
        ])
        contra_html = f"""
        <div style='background:#fff3f3;border-left:4px solid #cc2200;padding:12px;border-radius:6px'>
            <b style='color:#cc2200'>⚠️ Contradictions Detected:</b>
            <ul style='margin:8px 0 0 0;padding-left:20px'>{c_items}</ul>
        </div>
        """
    else:
        contra_html = """
        <div style='background:#f0fff0;border-left:4px solid #1a7a1a;padding:12px;border-radius:6px'>
            <b style='color:#1a7a1a'>✅ No contradictions detected across sources</b>
        </div>
        """

    # ── Temporal Flags ───────────────────────────────────────
    if result["temporal_flags"]:
        t_items = "".join([
            f"<li style='margin:6px 0;color:#7a5c00'>{t}</li>"
            for t in result["temporal_flags"]
        ])
        temporal_html = f"""
        <div style='background:#fffbea;border-left:4px solid #e6ac00;padding:12px;border-radius:6px'>
            <b style='color:#7a5c00'>⏰ Temporal Warnings:</b>
            <ul style='margin:8px 0 0 0;padding-left:20px'>{t_items}</ul>
        </div>
        """
    else:
        temporal_html = """
        <div style='background:#f0fff0;border-left:4px solid #1a7a1a;padding:12px;border-radius:6px'>
            <b style='color:#1a7a1a'>✅ All retrieved sources within recency threshold</b>
        </div>
        """

    return (
        confidence_html,
        answer_html,
        subquery_html,
        sources_html,
        contra_html,
        temporal_html
    )


# ── Build Gradio Interface ───────────────────────────────────
with gr.Blocks(
    title="AgentRAG",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container { max-width: 960px; margin: auto; }
        #title { text-align: center; margin-bottom: 4px; }
        #subtitle { text-align: center; color: #666; margin-bottom: 20px; }
    """
) as demo:

    # Header
    gr.HTML("<h1 id='title'>🔬 AgentRAG</h1>")
    gr.HTML("""
        <p id='subtitle'>
        Multi-Agent Retrieval-Augmented Generation
        with Self-Correcting Query Reformulation &amp;
        Confidence-Calibrated Responses
        </p>
    """)

    # Input
    with gr.Row():
        query_input = gr.Textbox(
            label="Medical Question",
            placeholder="e.g. What are the effects of metformin in elderly patients with type 2 diabetes?",
            lines=2,
            scale=5
        )
        submit_btn = gr.Button("🔍 Ask AgentRAG", variant="primary", scale=1)

    # Example questions
    gr.Examples(
        examples=[
            ["What are the treatment options for Type 2 diabetes?"],
            ["Is insulin therapy effective for elderly diabetic patients?"],
            ["What are the cardiovascular risks associated with diabetes medications?"],
            ["How does blood glucose control affect diabetes complications?"],
        ],
        inputs=query_input,
        label="Example Questions"
    )

    gr.HTML("<hr style='margin:20px 0'>")

    # Output panels
    with gr.Row():
        with gr.Column(scale=1):
            confidence_out = gr.HTML(label="Confidence Label")
        with gr.Column(scale=2):
            subquery_out = gr.HTML(label="Query Decomposition")

    answer_out = gr.HTML(label="Answer")

    with gr.Row():
        contra_out  = gr.HTML(label="Contradiction Detection")
        temporal_out = gr.HTML(label="Temporal Awareness")

    sources_out = gr.HTML(label="Retrieved Sources")

    # Footer
    gr.HTML("""
        <hr style='margin:20px 0'>
        <p style='text-align:center;color:#aaa;font-size:12px'>
        AgentRAG | Knowledge Base: PubMedQA |
        Agents: Query · Retrieval · Reasoning · Critic
        </p>
    """)

    # Wire up
    submit_btn.click(
        fn=process_query,
        inputs=query_input,
        outputs=[
            confidence_out,
            answer_out,
            subquery_out,
            sources_out,
            contra_out,
            temporal_out
        ]
    )

    query_input.submit(
        fn=process_query,
        inputs=query_input,
        outputs=[
            confidence_out,
            answer_out,
            subquery_out,
            sources_out,
            contra_out,
            temporal_out
        ]
    )


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("AgentRAG UI Starting...")
    print("Open your browser at: http://127.0.0.1:7860")
    print("=" * 50 + "\n")
    demo.launch(share=False)