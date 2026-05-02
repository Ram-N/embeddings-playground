
# 1) Product idea (simple, cohesive)

**App name:**
**“Embedding Playground”**

**Core idea:**
One interface, four tabs:

1. Word Explorer
2. Analogies
3. Sentence Similarity
4. Visualization

Everything revolves around:

> *“Meaning = position in space”*

---

# 2) UI Structure (Gradio Tabs)

You’ll use:

```python
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Word Explorer"):
            ...
```

---

# 3) Recommended model (for Spaces)

Skip raw GloVe here. Use a lightweight Hugging Face model:

* `sentence-transformers/all-MiniLM-L6-v2`

Why:

* small (~90MB)
* fast
* works for words + sentences
* no manual loading headaches

---

# 4) Full starter code (clean + modular)

This is a strong base. You or an agent can extend it.

```python
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------- Core helpers ----------

def embed(text):
    return model.encode(text)


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- 1. WORD EXPLORER ----------

def word_explorer(word):
    vec = embed(word)

    # Compare against small curated vocab (faster than full scan)
    vocab = ["dog", "cat", "pizza", "burger", "teacher", "student",
             "car", "bus", "doctor", "nurse", "music", "song"]

    sims = [(w, cosine(vec, embed(w))) for w in vocab]
    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    neighbors = [w for w, _ in sims[:5]]

    return str(vec[:10]), ", ".join(neighbors)


# ---------- 2. ANALOGY ----------

def analogy(a, b, c):
    va, vb, vc = embed(a), embed(b), embed(c)
    target = vb - va + vc

    vocab = ["king", "queen", "man", "woman", "teacher", "doctor",
             "engineer", "artist", "pilot", "chef", "student"]

    sims = [(w, cosine(target, embed(w))) for w in vocab]
    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    return ", ".join([w for w, _ in sims[:5]])


# ---------- 3. SENTENCE SIMILARITY ----------

def sentence_similarity(s1, s2):
    v1, v2 = embed(s1), embed(s2)
    sim = cosine(v1, v2)
    return f"Similarity: {sim:.3f}"


# ---------- 4. VISUALIZATION ----------

def visualize(words_text):
    words = [w.strip() for w in words_text.split(",")]
    vecs = [embed(w) for w in words]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vecs)

    plt.figure()
    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.text(reduced[i, 0], reduced[i, 1], word)

    return plt


# ---------- UI ----------

with gr.Blocks(title="Embedding Playground") as demo:

    gr.Markdown("# Embedding Playground")
    gr.Markdown("Explore how AI represents meaning as vectors")

    with gr.Tabs():

        # --- TAB 1 ---
        with gr.Tab("Word Explorer"):
            word_input = gr.Textbox(label="Enter a word")
            vec_output = gr.Textbox(label="Vector (first 10 dims)")
            neighbors_output = gr.Textbox(label="Closest words")

            btn = gr.Button("Explore")
            btn.click(word_explorer, inputs=word_input,
                      outputs=[vec_output, neighbors_output])

        # --- TAB 2 ---
        with gr.Tab("Analogies"):
            a = gr.Textbox(label="A")
            b = gr.Textbox(label="B")
            c = gr.Textbox(label="C")

            analogy_output = gr.Textbox(label="Result")

            btn2 = gr.Button("Solve Analogy")
            btn2.click(analogy, inputs=[a, b, c],
                       outputs=analogy_output)

        # --- TAB 3 ---
        with gr.Tab("Sentence Similarity"):
            s1 = gr.Textbox(label="Sentence 1")
            s2 = gr.Textbox(label="Sentence 2")

            sim_output = gr.Textbox(label="Similarity")

            btn3 = gr.Button("Compare")
            btn3.click(sentence_similarity,
                       inputs=[s1, s2],
                       outputs=sim_output)

        # --- TAB 4 ---
        with gr.Tab("Visualization"):
            words = gr.Textbox(
                label="Enter words (comma separated)",
                placeholder="cat, dog, pizza, car"
            )

            plot_output = gr.Plot()

            btn4 = gr.Button("Plot")
            btn4.click(visualize, inputs=words,
                       outputs=plot_output)


demo.launch()
```

---

# 5) How to deploy on Hugging Face Spaces

Create a Space:

* SDK: **Gradio**

Add:

### `requirements.txt`

```
gradio
sentence-transformers
scikit-learn
matplotlib
```

That’s it. It will auto-deploy.

---

# 6) Design improvements (what to tell your coding agent)

Ask for:

### A) Better UX

* Add examples under each tab
* Add “Try this:” suggestions
* Add loading spinner

---

### B) Teaching overlays

* Small explanation box per tab:

  * “Why are these words close?”
  * “What does this analogy mean?”

---

### C) Performance upgrade

* Cache embeddings:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed(text):
    return model.encode(text)
```

---

### D) Better visualization

* Color clusters
* Animate transitions (advanced)

---

# 7) Key teaching advantage of this setup

This UI lets you:

* live demo
* let students experiment
* reinforce intuition visually

---
