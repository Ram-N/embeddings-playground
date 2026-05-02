import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gensim.downloader as api

# Load both models at startup
print("Loading GloVe...")
glove = api.load("glove-wiki-gigaword-100")

print("Loading Sentence Transformer...")
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Both models ready.")


# ---------- Core helpers ----------

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_st(text):
    return st_model.encode(text)


# ---------- 1. WORD EXPLORER ----------

ST_VOCAB = ["dog", "cat", "pizza", "burger", "teacher", "student",
            "car", "bus", "doctor", "nurse", "music", "song"]

def word_explorer(word, model_choice):
    word = word.lower().strip()

    if model_choice == "GloVe":
        if word not in glove:
            return "Word not in GloVe vocabulary", ""
        vec = glove[word]
        neighbors = [w for w, _ in glove.most_similar(word, topn=5)]
    else:
        vec = embed_st(word)
        sims = [(w, cosine(vec, embed_st(w))) for w in ST_VOCAB]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        neighbors = [w for w, _ in sims[:5]]

    return str(vec[:10]), ", ".join(neighbors)


# ---------- 2. ANALOGY ----------

ST_ANALOGY_VOCAB = ["king", "queen", "man", "woman", "teacher", "doctor",
                    "engineer", "artist", "pilot", "chef", "student"]

def analogy(a, b, c, model_choice):
    a, b, c = a.lower().strip(), b.lower().strip(), c.lower().strip()

    if model_choice == "GloVe":
        missing = [w for w in [a, b, c] if w not in glove]
        if missing:
            return f"Not in GloVe vocabulary: {', '.join(missing)}", ""
        result = glove[b] - glove[a] + glove[c]
        neighbors = [w for w, _ in glove.most_similar(positive=[b, c], negative=[a], topn=5)]
    else:
        va, vb, vc = embed_st(a), embed_st(b), embed_st(c)
        result = vb - va + vc
        sims = [(w, cosine(result, embed_st(w))) for w in ST_ANALOGY_VOCAB]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        neighbors = [w for w, _ in sims[:5] if w not in [a, b, c]]

    return str(result[:10]), ", ".join(neighbors)


# ---------- 3. SENTENCE SIMILARITY ----------

def sentence_similarity(s1, s2, model_choice):
    if model_choice == "GloVe":
        def sentence_vec(sentence):
            words = sentence.lower().split()
            vecs = [glove[w] for w in words if w in glove]
            return np.mean(vecs, axis=0) if vecs else None

        v1, v2 = sentence_vec(s1), sentence_vec(s2)
        if v1 is None or v2 is None:
            return "Could not compute: no recognisable words found"
        note = " (GloVe: averaging word vectors)"
    else:
        v1, v2 = embed_st(s1), embed_st(s2)
        note = " (Sentence Transformer: full sentence encoding)"

    sim = cosine(v1, v2)
    return f"Similarity: {sim:.3f}{note}"


# ---------- 4. VISUALIZATION ----------

WORD_SETS = {
    "Semantic clusters": {
        "words":  ["tiger", "elephant", "cobra", "peacock", "biryani", "dosa", "samosa", "doctor", "engineer", "farmer"],
        "groups": ["animal", "animal", "animal", "animal", "food", "food", "food", "profession", "profession", "profession"],
    },
    "Gender pairs": {
        "words":  ["raja", "rani", "bhai", "behan", "actor", "actress", "uncle", "aunty"],
        "groups": ["male", "female", "male", "female", "male", "female", "male", "female"],
    },
    "Countries + Capitals": {
        "words":  ["india", "delhi", "china", "beijing", "japan", "tokyo", "france", "paris", "brazil", "brasilia"],
        "groups": ["country", "capital", "country", "capital", "country", "capital", "country", "capital", "country", "capital"],
    },
    "Antonyms": {
        "words":  ["hot", "cold", "happy", "sad", "big", "small", "fast", "slow"],
        "groups": ["temp", "temp", "mood", "mood", "size", "size", "speed", "speed"],
    },
    "Abstract vs Concrete": {
        "words":  ["dharma", "karma", "freedom", "justice", "chapati", "tabla", "cricket", "hope"],
        "groups": ["abstract", "abstract", "abstract", "abstract", "concrete", "concrete", "concrete", "abstract"],
    },
    "Custom": None,
}


def _assign_colors(groups):
    """Map unique group names to distinct colours."""
    unique = list(dict.fromkeys(groups))  # preserve order, deduplicate
    palette = cm.get_cmap("tab10", len(unique))
    color_map = {g: palette(i) for i, g in enumerate(unique)}
    return [color_map[g] for g in groups], color_map


def _scatter(ax, coords, words, colors, title):
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, zorder=2)
    for i, word in enumerate(words):
        ax.annotate(word, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_title(title)
    ax.axhline(0, color="lightgrey", linewidth=0.5)
    ax.axvline(0, color="lightgrey", linewidth=0.5)


def _make_legend(ax, color_map):
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markersize=8, label=group)
        for group, color in color_map.items()
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)


def visualize(words_text, model_choice, selected_set):
    if selected_set != "Custom":
        entry = WORD_SETS[selected_set]
        words = entry["words"]
        groups = entry["groups"]
    else:
        words = [w.strip().lower() for w in words_text.split(",") if w.strip()]
        groups = None

    # Get vectors
    if model_choice == "GloVe":
        if groups is not None:
            valid = [(w, g, glove[w]) for w, g in zip(words, groups) if w in glove]
            skipped = [w for w, g in zip(words, groups) if w not in glove]
        else:
            valid = [(w, None, glove[w]) for w in words if w in glove]
            skipped = [w for w in words if w not in glove]

        if len(valid) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough words found in GloVe vocabulary.\nTry switching to Sentence Transformers.",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        words = [v[0] for v in valid]
        groups = [v[1] for v in valid] if groups is not None else None
        vecs = np.array([v[2] for v in valid])
    else:
        vecs = np.array([embed_st(w) for w in words])
        skipped = []

    # Dimensionality reduction
    pca_2d = PCA(n_components=2).fit_transform(vecs)
    perplexity = min(30, len(words) - 1)
    tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(vecs)

    # Colours
    if groups is not None:
        colors, color_map = _assign_colors(groups)
    else:
        colors = ["steelblue"] * len(words)
        color_map = None

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _scatter(ax1, pca_2d, words, colors, title=f"PCA  ({model_choice})")
    _scatter(ax2, tsne_2d, words, colors, title=f"t-SNE  ({model_choice})")

    if color_map:
        _make_legend(ax2, color_map)

    if skipped:
        fig.text(0.5, 0.01, f"Skipped (not in GloVe): {', '.join(skipped)}",
                 ha="center", fontsize=8, color="grey")

    fig.tight_layout()
    return fig


# ---------- UI ----------

with gr.Blocks(title="Embedding Playground") as demo:

    gr.Markdown("# Embedding Playground")
    gr.Markdown("Explore how AI represents meaning as vectors in space.")

    model_choice = gr.Radio(
        choices=["GloVe", "Sentence Transformers"],
        value="GloVe",
        label="Embedding Model",
        info="GloVe: classic word vectors trained on Wikipedia. "
             "Sentence Transformers: modern neural embeddings that handle full sentences."
    )

    with gr.Tabs():

        # --- TAB 1 ---
        with gr.Tab("Word Explorer"):
            gr.Markdown("Enter a word to see its vector and closest neighbours.")
            word_input = gr.Textbox(label="Enter a word")
            vec_output = gr.Textbox(label="Vector (first 10 dims)")
            neighbors_output = gr.Textbox(label="Closest words")
            btn = gr.Button("Explore")
            btn.click(word_explorer, inputs=[word_input, model_choice],
                      outputs=[vec_output, neighbors_output])

        # --- TAB 2 ---
        with gr.Tab("Analogies"):
            gr.Markdown("### A is to B as C is to ?")
            gr.Markdown("Vector arithmetic: **B − A + C** → find the closest word")
            a_in = gr.Textbox(label="A (starting point)")
            b_in = gr.Textbox(label="B (related to A)")
            c_in = gr.Textbox(label="C (new starting point)")
            analogy_vec = gr.Textbox(label="Result vector (first 10 dims)")
            analogy_out = gr.Textbox(label="Closest words")
            btn2 = gr.Button("Solve Analogy")
            btn2.click(analogy, inputs=[a_in, b_in, c_in, model_choice],
                       outputs=[analogy_vec, analogy_out])

        # --- TAB 3 ---
        with gr.Tab("Sentence Similarity"):
            gr.Markdown("Compare two sentences. Score ranges from 0 (unrelated) to 1 (identical meaning).")
            s1 = gr.Textbox(label="Sentence 1")
            s2 = gr.Textbox(label="Sentence 2")
            sim_output = gr.Textbox(label="Similarity")
            btn3 = gr.Button("Compare")
            btn3.click(sentence_similarity, inputs=[s1, s2, model_choice],
                       outputs=sim_output)

        # --- TAB 4 ---
        with gr.Tab("Visualization"):
            gr.Markdown("Words are plotted in 2D space — left using PCA, right using t-SNE.")
            gr.Markdown(
                "**PCA**: distances between clusters are meaningful. "
                "**t-SNE**: clusters are visually clearer, but distances *between* clusters are not meaningful."
            )
            set_dropdown = gr.Dropdown(
                choices=list(WORD_SETS.keys()),
                value="Semantic clusters",
                label="Word set — select one option from the dropdown, then press Plot"
            )
            custom_words = gr.Textbox(
                label="Custom words (comma separated — only used when 'Custom' is selected above)",
                placeholder="e.g. moon, star, sun, cloud, rain"
            )
            btn4 = gr.Button("Plot")
            plot_output = gr.Plot()

            set_dropdown.change(
                fn=lambda s: "" if s != "Custom" else gr.update(),
                inputs=set_dropdown,
                outputs=custom_words
            )

            btn4.click(visualize, inputs=[custom_words, model_choice, set_dropdown],
                       outputs=plot_output)

demo.launch()
