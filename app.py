import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# ---------- Presets ----------

WORD_PRESETS = {
    "tiger":   "tiger",
    "biryani": "biryani",
    "doctor":  "doctor",
    "dharma":  "dharma",
    "cricket": "cricket",
}

ANALOGY_PRESETS = {
    "Countries → Capitals  (india : delhi :: france : ?)": ("india",  "delhi",  "france"),
    "Gender  (man : king :: woman : ?)":                   ("man",    "king",   "woman"),
    "Institutions  (school : teacher :: hospital : ?)":    ("school", "teacher","hospital"),
    "Nature  (day : sun :: night : ?)":                    ("day",    "sun",    "night"),
}

SENTENCE_PRESETS = {
    "Similar meaning":               ("I love eating biryani",  "I enjoy having dosa"),
    "Negation — tricky for GloVe":   ("I am happy",             "I am not happy"),
    "Same word, different meaning":  ("The bat flew at night",  "He swung the bat"),
    "Unrelated":                     ("The train is very fast", "My grandmother makes great chai"),
}


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


def _auto_cluster(vecs):
    """Pick the best k via silhouette score and return group labels."""
    n = len(vecs)
    max_k = min(6, n - 1)
    if max_k < 2:
        return ["cluster_0"] * n

    best_score, best_labels = -1, None
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(vecs)
        score = silhouette_score(vecs, labels)
        if score > best_score:
            best_score, best_labels = score, labels

    return [f"cluster_{l}" for l in best_labels]


def _assign_colors(groups):
    """Map unique group names to distinct colours."""
    unique = list(dict.fromkeys(groups))  # preserve order, deduplicate
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(unique)}
    return [color_map[g] for g in groups], color_map


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
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough words found in GloVe vocabulary.<br>Try switching to Sentence Transformers.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14)
            )
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

    # Colours — auto-cluster custom words; use predefined groups for presets
    if groups is None:
        groups = _auto_cluster(np.array(vecs))
    _, color_map = _assign_colors(groups)

    # Build Plotly subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"PCA  ({model_choice})", f"t-SNE  ({model_choice})"])

    unique_groups = list(dict.fromkeys(groups))
    for grp in unique_groups:
        indices = [j for j, g in enumerate(groups) if g == grp]
        color = color_map[grp]

        fig.add_trace(go.Scatter(
            x=pca_2d[indices, 0], y=pca_2d[indices, 1],
            mode="markers+text",
            text=[words[j] for j in indices],
            textposition="top center",
            marker=dict(color=color, size=9),
            name=grp,
            legendgroup=grp,
            showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=tsne_2d[indices, 0], y=tsne_2d[indices, 1],
            mode="markers+text",
            text=[words[j] for j in indices],
            textposition="top center",
            marker=dict(color=color, size=9),
            name=grp,
            legendgroup=grp,
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(height=520, margin=dict(t=60, b=40))

    if skipped:
        fig.add_annotation(
            text=f"Skipped (not in GloVe): {', '.join(skipped)}",
            xref="paper", yref="paper", x=0.5, y=-0.05,
            showarrow=False, font=dict(size=10, color="grey")
        )

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
            gr.Markdown("""
**Try these Indian words** — copy any one word into the box below and hit Explore:

ratnam · kamal · modi · nehru · gandhi · sholay · chai · adda · bilkul · yaar · jugaad · desi · arre · timepass · biryani · dosa · lassi · namaste · dharma · karma · cricket · monsoon · rickshaw · mumbai · delhi · bollywood · masala · jugnu · tapori · bakwaas · jadoo · zindagi · azaadi · shaayar · filmi · chutzpah · nawab · zamindar · memsaab · babu · paisa

*(Note: each entry above is a single word. "Cutting chai" and "adda" work best with Sentence Transformers since GloVe may not have them in its vocabulary.)*
""")
            word_mode = gr.Radio(["Pre-set", "Custom"], value="Custom", label="Input mode")
            word_preset = gr.Dropdown(
                choices=list(WORD_PRESETS.keys()), value="tiger",
                label="Preset word", interactive=False
            )
            word_input = gr.Textbox(label="Custom word", value="ratnam", interactive=True)
            vec_output = gr.Textbox(label="Vector (first 10 dims)")
            neighbors_output = gr.Textbox(label="Closest words")
            btn = gr.Button("Explore", variant="primary")

            def fill_word(preset):
                return WORD_PRESETS[preset]

            def toggle_word_mode(mode):
                if mode == "Pre-set":
                    return gr.update(interactive=True), gr.update(interactive=False)
                return gr.update(interactive=False), gr.update(interactive=True)

            word_mode.change(toggle_word_mode, inputs=word_mode, outputs=[word_preset, word_input])
            word_preset.change(fill_word, inputs=word_preset, outputs=word_input)
            btn.click(word_explorer, inputs=[word_input, model_choice],
                      outputs=[vec_output, neighbors_output])

            gr.Markdown("""
---
### What am I looking at?

**The vector** is a list of numbers — the word's "address" in a high-dimensional space of meaning.
The model learned these numbers by reading billions of words and noticing which words tend to appear together.
Words used in similar contexts end up with similar vectors — that's the core idea behind word embeddings.

**Why only 10 numbers?**
The full vector has many more dimensions — **GloVe stores 100 numbers per word**, while
**Sentence Transformers stores 384**. Showing all of them would just be a wall of digits.
The first 10 give you a feel for what a vector looks like; no single number has an obvious human-readable meaning.

**What do the numbers actually mean?**
Each dimension loosely captures some latent feature of language (context, topic, grammar role, etc.),
but the dimensions are not individually labelled. What matters is the *overall pattern*:
two words with similar vectors are semantically close — you can see this in the "Closest words" output.

**Why do the closest words make sense?**
The model computes the cosine similarity between the query vector and every word in its vocabulary,
then returns the top matches. Words that appear in similar sentences — like *doctor*, *nurse*, *physician* —
end up near each other in vector space.
""")


        # --- TAB 2 ---
        with gr.Tab("Analogies"):
            gr.Markdown("### A is to B as C is to ?")
            gr.Markdown("Vector arithmetic: **B − A + C** → find the closest word")
            analogy_mode = gr.Radio(["Pre-set", "Custom"], value="Pre-set", label="Input mode")
            analogy_preset = gr.Dropdown(
                choices=list(ANALOGY_PRESETS.keys()),
                value=list(ANALOGY_PRESETS.keys())[0],
                label="Preset analogy"
            )
            _default_a, _default_b, _default_c = ANALOGY_PRESETS[list(ANALOGY_PRESETS.keys())[0]]
            a_in = gr.Textbox(label="A (starting point)", value=_default_a, interactive=False)
            b_in = gr.Textbox(label="B (related to A)", value=_default_b, interactive=False)
            c_in = gr.Textbox(label="C (new starting point)", value=_default_c, interactive=False)
            analogy_vec = gr.Textbox(label="Result vector (first 10 dims)")
            analogy_out = gr.Textbox(label="Closest words")
            btn2 = gr.Button("Solve Analogy", variant="primary")

            def fill_analogy(preset):
                a, b, c = ANALOGY_PRESETS[preset]
                return a, b, c

            def toggle_analogy_mode(mode):
                if mode == "Pre-set":
                    return gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
                return gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

            analogy_mode.change(toggle_analogy_mode, inputs=analogy_mode, outputs=[analogy_preset, a_in, b_in, c_in])
            analogy_preset.change(fill_analogy, inputs=analogy_preset, outputs=[a_in, b_in, c_in])
            btn2.click(analogy, inputs=[a_in, b_in, c_in, model_choice],
                       outputs=[analogy_vec, analogy_out])

        # --- TAB 3 ---
        with gr.Tab("Sentence Similarity"):
            gr.Markdown("Compare two sentences. Score ranges from 0 (unrelated) to 1 (identical meaning).")
            sent_mode = gr.Radio(["Pre-set", "Custom"], value="Pre-set", label="Input mode")
            sent_preset = gr.Dropdown(
                choices=list(SENTENCE_PRESETS.keys()),
                value="Similar meaning",
                label="Preset sentence pair"
            )
            _default_s1, _default_s2 = SENTENCE_PRESETS["Similar meaning"]
            s1 = gr.Textbox(label="Sentence 1", value=_default_s1, interactive=False)
            s2 = gr.Textbox(label="Sentence 2", value=_default_s2, interactive=False)
            sim_output = gr.Textbox(label="Similarity")
            btn3 = gr.Button("Compare", variant="primary")

            def fill_sentences(preset):
                s1v, s2v = SENTENCE_PRESETS[preset]
                return s1v, s2v

            def toggle_sent_mode(mode):
                if mode == "Pre-set":
                    return gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)
                return gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=True)

            sent_mode.change(toggle_sent_mode, inputs=sent_mode, outputs=[sent_preset, s1, s2])
            sent_preset.change(fill_sentences, inputs=sent_preset, outputs=[s1, s2])
            btn3.click(sentence_similarity, inputs=[s1, s2, model_choice],
                       outputs=sim_output)

        # --- TAB 4 ---
        with gr.Tab("Visualization"):
            gr.Markdown("Words are plotted in 2D space — left using PCA, right using t-SNE.")
            gr.Markdown(
                "**PCA**: distances between clusters are meaningful. "
                "**t-SNE**: clusters are visually clearer, but distances *between* clusters are not meaningful."
            )
            viz_mode = gr.Radio(["Pre-set", "Custom"], value="Pre-set", label="Input mode")
            set_dropdown = gr.Dropdown(
                choices=list(WORD_SETS.keys()),
                value="Semantic clusters",
                label="Word set"
            )
            custom_words = gr.Textbox(
                label="Custom words (comma separated)",
                placeholder="e.g. moon, star, sun, cloud, rain",
                interactive=False
            )
            btn4 = gr.Button("Plot", variant="primary")
            plot_output = gr.Plot()

            def toggle_viz_mode(mode):
                if mode == "Pre-set":
                    return gr.update(interactive=True), gr.update(interactive=False)
                return gr.update(interactive=False, value="Custom"), gr.update(interactive=True)

            viz_mode.change(toggle_viz_mode, inputs=viz_mode, outputs=[set_dropdown, custom_words])

            btn4.click(visualize, inputs=[custom_words, model_choice, set_dropdown],
                       outputs=plot_output)

demo.launch()
