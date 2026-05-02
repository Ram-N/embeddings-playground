# Lesson: Having Fun with Embeddings (30 mins)

## Goal

Students should walk away with one core idea:

> *Meaning = position in space*

---

## 0–3 min — Hook (make it intuitive)

Start with a question:

* "Is *cat* closer to *dog* or *car*?"

Then say:

* "Computers don't understand meaning. They only understand numbers."
* "So what if we turn words into coordinates?"

**Key line:**

> "Embeddings are just GPS coordinates for meaning."

---

## 3–8 min — First demo (single word)

Open the Gradio app. Use the **Word Explorer** tab. Make sure the model toggle is set to **GloVe**.

### Do:

* Input: `cat`
* Show:
  * Vector (just first few numbers)
  * Nearest words: `dog, kitten, pet`

### Ask:

* "Why are these words close?"
* "What would happen with 'pizza'?"

Try:

* `pizza` → `pasta, sandwich`

**Instructor note:** Don't linger on the vector numbers — students find them intimidating. Point at them briefly and say "that's just a list of coordinates, like latitude and longitude" and move on to the nearest words, which are immediately intuitive.

**Key takeaway:**

> "Words that appear in similar contexts end up close together."

---

## 8–13 min — Neighbourhood intuition

### Activity:

Ask students to suggest words. Try a few live:

* `school`
* `music`
* `doctor`

Show nearest neighbours each time.

### Push their thinking:

* "Is this about spelling?"
* "Or meaning?"

**Instructor note:** The answer they need to arrive at is *meaning, not spelling*. Spell out the contrast: `cat` and `car` look similar but end up far apart. `doctor` and `physician` look nothing alike but end up close. Let a student make this observation if possible — it lands better coming from them.

**Key takeaway:**

> "The model never learned definitions. It learned patterns."

---

## 13–18 min — Analogies (the "magic" moment)

Introduce:

> "We can *do math* with meaning."

Switch to the **Analogies** tab. Keep the model on **GloVe** — it handles classic analogies well and is faster.

### Examples to run:

| A | B | C | Expected result |
|---|---|---|-----------------|
| france | paris | italy | rome |
| school | teacher | hospital | doctor |
| day | sun | night | moon |

Let students suggest one of their own.

### Ask:

* "Why does this work?"

**Explain simply:**

* Direction = relationship
* `paris - france` captures the idea of "capital city of"
* Apply that direction to `italy` and you land near `rome`

**Instructor note:** Point out the Countries + Capitals set in the Visualization tab later as a visual proof of this. In the PCA plot, each country and its capital appear as a pair with a consistent offset — you can literally see the "capital city of" direction as a repeated arrow. Mention this now so students know it's coming.

**Key takeaway:**

> "Relationships between words are consistent directions in space."

---

## 18–23 min — Sentence embeddings

Switch to the **Sentence Similarity** tab. Switch the model toggle to **Sentence Transformers**.

### Show pair 1 (similar meaning):

* "I love eating pizza"
* "I enjoy pasta"

### Show pair 2 (unrelated):

* "I love eating pizza"
* "The car is very fast"

### Ask:

* "Which pair is closer? Why?"

Then switch the toggle to **GloVe** and run the same pairs.

**Teaching moment — bag of words vs. full encoding:**

* GloVe has no concept of a sentence. It averages the word vectors together — order and grammar are thrown away.
* That means "The dog chased the cat" and "The cat chased the dog" get nearly identical scores — same words, same average.
* Sentence Transformers read the whole sequence, so word order and grammar actually matter.

**Instructor note — the demo that makes this concrete:** Type `"I am happy"` and `"I am not happy"`. With GloVe the score will be very high (nearly identical vectors). With Sentence Transformers the score drops noticeably. Ask students: *"Which model is more useful for understanding what someone actually said?"* The answer is obvious, and it sets up why modern models matter.

**Key takeaway:**

> "We can represent *ideas*, not just words — and how we do it matters."

---

## 23–27 min — Ambiguity (strong moment)

Still in **Sentence Similarity** tab.

### Run:

* "The bat flew at night"
* "He swung the bat"

Show that the similarity score is lower than students might expect — same word, very different meaning.

### Ask:

* "Same word. Different meaning. What changed?"

**Explain:**

* In GloVe, `bat` has one fixed vector forever — animal and cricket bat are collapsed into one point. It can't handle ambiguity.
* Sentence Transformers use the surrounding words as context to shift the representation — `flew at night` pulls the vector toward the animal sense; `swung` pulls it toward the sports equipment sense.

**Instructor note:** This is the moment to say *"this is exactly the problem that GPT and modern LLMs were built to solve."* Every word's meaning is recalculated based on context, every single time. That's why they're so much more powerful — and so much more expensive to run.

**Key takeaway:**

> "Meaning is not in the word. It's in the context."

---

## 27–30 min — Wrap + big picture

### Recap (quick-fire):

* Words → vectors
* Similar meaning → close in space
* Analogies → relationships are directions
* Sentences → combined and contextual meaning
* Context → shifts meaning

### Final question:

* "Why does any of this matter?"

Guide them toward:

* **Search** — find results by meaning, not just keywords
* **Recommendations** — "you liked X, here's something nearby in space"
* **Chatbots** — understanding what you're asking, not just matching words
* **Translation** — meaning is similar across languages in the same space

**Closing line:**

> "Embeddings are the foundation of how modern AI understands language."

---

## Visualization tab — instructor guide

Use this tab as a visual payoff after the analogies and similarity sections. It works best as a live "let's see it" moment rather than a structured demo.

### Recommended sequence:

**1. Start with Semantic clusters**

Select from the dropdown. Point out that animals, foods, and professions each form their own cluster — especially clear in t-SNE. Ask: *"Did we tell the model what a food word is? Or an animal?"* No — it figured it out from context.

**2. Switch to Antonyms — the counterintuitive moment**

This is the strongest teaching demo in the visualization tab. Students will expect `hot` and `cold` to be far apart. They're not — they cluster together.

**Emphasize:** The model learned from context, not definitions. `Hot` and `cold` appear in almost identical sentences ("it was very ___", "the weather is ___"). The model has no idea they mean opposite things — it only knows they live in the same neighbourhood of text.

Ask: *"What does this tell us about the limits of these models?"*

**3. Countries + Capitals — visual proof of analogy directions**

Point out that each country and its capital appear as a pair. In the PCA plot, the offset between country and capital is roughly consistent across all pairs. This is the geometric version of what you showed in the Analogies tab — the "capital city of" direction is real and visible.

**4. Hindi words — the training data moment**

Select **Gender pairs** and switch the model to **GloVe**. The words `raja`, `rani`, `bhai`, `behan` will be skipped (the app notes this at the bottom of the plot). Switch to **Sentence Transformers** and they work.

**Emphasize:** GloVe was trained on English Wikipedia and news. It simply never saw these words enough to learn them. This is not a flaw in the concept — it's a flaw in the training data. Ask: *"If we trained on Hindi text, what would change?"* This opens the door to multilingual models.

**Instructor note — PCA vs t-SNE:** Students will ask why the two plots look different. The one-line answer: *"PCA is like taking a photo from a fixed angle — distances are preserved but some clusters may overlap. t-SNE rearranges everything to make clusters as clear as possible, but the distances between clusters are now meaningless."* You only trust t-SNE for asking "are these words in the same neighbourhood?" — not for asking "how far apart are these two clusters?"

---

## Optional extensions

* Mini challenge: find an analogy that *breaks* — where the model gets it wrong — and discuss why
* Ask: *"What's missing from this picture?"* — lead into limitations (bias in training data, rare words, multilinguality)
* Try the Custom word set in Visualization with student-suggested words — let them make predictions before hitting Plot
