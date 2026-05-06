# Embedding Atlas

Embedding Atlas is a tool for visually exploring large collections of text by mapping them onto a 2D canvas — like a geographic map, but for meaning.

## What It Does (Step by Step)

When you run a command like:

```bash
embedding-atlas stanfordnlp/imdb --text "text" --sample 1000
```

Here's what happens under the hood:

### 1. Download the dataset
It fetches 1000 rows from the IMDB dataset on Hugging Face Hub. Each row is a movie review with a sentiment label (positive/negative).

### 2. Load a pre-trained embedding model
It downloads `all-MiniLM-L6-v2` — a small, fast sentence-transformer model (~90MB). This model already knows how language works; **nothing is trained or fine-tuned here**. The weights are cached locally after the first download.

### 3. Run inference (embedding)
Each review is passed through the model, which outputs a vector of **384 numbers**. This vector is the "embedding" — a mathematical representation of the text's meaning. Similar texts produce similar vectors; unrelated texts produce different ones.

This is the most compute-intensive step, but even on CPU it's fast for 1000 samples (~5 seconds).

### 4. Run UMAP
UMAP (Uniform Manifold Approximation and Projection) compresses those 384-dimensional vectors down to just **2 dimensions** so they can be plotted on a flat canvas. It tries to preserve the neighborhood relationships — points that were close in 384D stay close in 2D.

### 5. Open the browser
Embedding Atlas starts a local web server and opens an interactive visualization.

---

## What Are We Visualizing?

The 2D map shows **semantic space** — where each dot is a piece of text, and its position reflects its meaning.

- **Nearby dots** = texts with similar meaning or topic
- **Distant dots** = texts that are semantically different
- **Clusters** = groups of texts that share a theme, topic, or style

For the IMDB dataset, you'd expect to see clusters form around:
- Genre (action, romance, horror...)
- Sentiment (strongly positive vs. strongly negative reviews)
- Writing style (short punchy reviews vs. long detailed ones)

This is powerful because **you didn't define any of these categories** — the model's understanding of language created the structure automatically.

---

## Key Concepts

| Term | What it means |
|------|--------------|
| **Embedding** | A list of numbers representing a text's meaning |
| **Vector** | Another word for that list of numbers |
| **384 dimensions** | The size of each embedding from `all-MiniLM-L6-v2` |
| **UMAP** | Algorithm that squashes high-dimensional data to 2D while preserving structure |
| **Semantic space** | The abstract "space" where meaning lives — similar things are nearby |
| **Sentence Transformer** | A model trained to produce useful embeddings for sentences/paragraphs |

---

## Running It

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Basic usage — compute embeddings on the fly
embedding-atlas stanfordnlp/imdb --text "text" --sample 1000

# Use a specific model
embedding-atlas stanfordnlp/imdb --text "text" --model "sentence-transformers/all-MiniLM-L6-v2"

# If you already have x/y coordinates (pre-computed), skip the embedding step
embedding-atlas your-dataset --x "embedding_x" --y "embedding_y"
```

---

## Why It's Not "Training"

A common point of confusion: this tool does **not** train a model. It uses an already-trained model purely for **inference** — feeding text in, getting vectors out. Think of it like using a calculator: you're not building the calculator, you're just using it.

---

## Hands-On Workshop: Bollywood Embeddings

For a structured classroom exercise, see **`bollywood_embeddings_workshop.ipynb`** in the root of this repo.

The notebook walks students through the same pipeline — embed → UMAP → cluster → explore — using 2,000 Bollywood movie plots as the dataset. Key additions over the basic Embedding Atlas flow:

- Uses `paraphrase-multilingual-MiniLM-L12-v2` (supports Hindi/Hinglish/English)
- Auto-clusters with KMeans (no genre labels) and compares discovered clusters to actual genres
- Interactive Plotly scatter — hover to read plot summaries
- "Try it yourself" cell: type any Hindi or English description and find the 5 nearest movies, with a star marker showing where your query lands on the map

**Open in Colab**: upload `bollywood_embeddings_workshop.ipynb` to [colab.research.google.com](https://colab.research.google.com) and run all cells. No local setup needed. Runs in ~8 minutes on free-tier CPU.
