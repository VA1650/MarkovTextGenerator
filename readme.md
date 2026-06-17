# 🧠 MarkovTextGenerator — N-Gram Language Model

A lightweight, probability-based language model implemented in pure Python, leveraging N-gram statistical analysis and Markov Chains.

## 🧠 Mathematical Core
The project models language through conditional probabilities. Instead of complex semantic embedding, the model calculates the probability of the next word based on a fixed-length window of previous tokens (history):

$$P(w_i | w_{i-(n-1)}, \dots, w_{i-1})$$

### Architectural Highlights
* **Context Scalability ($N$):** The `NGramLanguageModel` class is fully parameterized. Switching between bigrams ($N=2$), trigrams ($N=3$), or deeper contexts requires only a single variable change.
* **Fit/Generate Lifecycle:** Built with a Scikit-learn style interface. Tokenization and frequency distribution modeling (`.fit()`) are isolated from inference. Generation is optimized to $O(1)$ lookup time using hash tables (`defaultdict`).
* **Dead-end Handling:** Implements "back-off" logic. If the model reaches a state with no known successors, it performs a random walk to a valid state, preventing termination and ensuring continuous text generation.

## 📈 Context Comparison (Bigrams vs. Trigrams)
* **Bigrams ($N=2$):** 1-word context. High entropy, resulting in grammatically loose but highly variable sequences.
* **Trigrams ($N=3$):** 2-word context. Improved local coherence, preserving idioms and fundamental syntax of the source corpus.

## 🚀 Quick Start
No external dependencies required (Pure Python).
1. Place your training corpus in `text.txt`.
2. Run the generator:
```bash
python markov_generator.py
```
3. The generated output will be stored in generated_text.txt
