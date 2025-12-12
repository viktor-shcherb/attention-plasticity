# Key Pair Sampling for Attention Plasticity

This document describes the sampling procedure used to estimate **attention plasticity** for a single attention head, based on sampled **key pairs** within transformer sequences.

---

## 1. Background and Motivation

We are interested in quantifying how **flexibly** an attention head can prioritize one key over another as the query distribution changes over time. For a given timestep $ t $ and a pair of keys $ (k_1, k_2) $, we define an event

$$
A_t := \{ (q_t, k_1) > (q_t, k_2) \}
$$

where $ q_t $ is a random query vector at time $ t $, and $ (\cdot,\cdot) $ denotes the dot product.

Let

$$
p_t := \mathbb{P}(A_t),
$$

and define the attention plasticity for that **key pair** at time $ t $ as

$$
\mathrm{AP}_t(k_1, k_2) := 4 \, \mathrm{Var}(\mathrm{Ber}(A_t)) = 4p_t(1 - p_t).
$$

- $ \mathrm{AP}_t(k_1, k_2) $ is maximal when $ p_t = 0.5 $, meaning the head is highly sensitive to small changes in the query (it can flip preference between $ k_1 $ and $ k_2 $).
- $ \mathrm{AP}_t(k_1, k_2) $ is minimal when $ p_t \approx 0 $ or $ p_t \approx 1 $, meaning the head has a nearly fixed preference for one of the keys.

To obtain a **head-level measure** of attention plasticity at time $ t $, we average this quantity over many sampled key pairs:

$$
\mathrm{AP}_t^{\text{head}} = \mathbb{E}_{(k_1, k_2) \sim \mathcal{S}} \big[ \mathrm{AP}_t(k_1, k_2) \big]
$$

where $ \mathcal{S} $ is the distribution defined by our key pair sampling procedure.

The quality and interpretability of $ \mathrm{AP}_t^{\text{head}} $ heavily depend on how we sample pairs $ (k_1, k_2) $. This document specifies a sampling scheme designed to avoid positional artifacts and ensure good coverage.

---

## 2. Design Goals for Sampling

We want a distribution $ \mathcal{S} $ over key pairs with the following properties:

1. **Same-example constraint**  
   Both keys in a pair must come from the same input example (same sequence).

2. **Locality constraint**  
   The keys must be **near each other** in the sequence to avoid confounding with large positional differences. Concretely:

   $$
   |i - j| \le \Delta_b, \quad \Delta_b = \max\big(1, \lfloor 0.05 \cdot L_b \rfloor \big),
   $$

   where $ L_b $ is the length of example $ b $, and $ i, j $ are key positions.

3. **Uniform coverage over sequence positions**  
   We want key pairs to be distributed approximately uniformly across the sequence length, so that plasticity estimates are not dominated by, e.g., early or late positions.

4. **No self-pairs**  
   Never choose the same key twice in one pair:

   $$
   i \neq j.
   $$

5. **No directional bias**  
   There should be no built-in preference for “earlier” vs “later” key in the pair when interpreting $ (k_1, k_2) $.

---

## 3. Notation

- We have a set of examples indexed by $ b = 0, 1, \dots, B-1 $.
- Example $ b $ has sequence length $ L_b $ (number of tokens/keys).
- Keys are indexed by $ (b, t) $, where $ t \in \{0, \dots, L_b - 1\} $.
- We want to sample $ N $ key pairs in total.

---

## 4. Sampling Procedure

### 4.1 High-level idea

1. First, choose an **example** $ b $ with probability proportional to its length $ L_b $. This makes each **position** in the dataset equally likely to be chosen as an anchor.
2. Within that example, sample an **anchor position** $ i $ uniformly across its length.
3. Sample a **neighbor position** $ j $ within a local window around $ i $, constrained to $ |i - j| \le \Delta_b $ and $ j \neq i $.
4. Randomly assign which of $ (i, j) $ is $ k_1 $ and which is $ k_2 $.

Repeating this gives us a collection of key pairs spread roughly uniformly over the sequence, while satisfying the local-distance and same-example constraints.

---

### 4.2 Formal description

Let

$$
\Delta_b = \max\big(1, \lfloor 0.05 \cdot L_b \rfloor \big).
$$

Define a probability distribution over examples:

$$
P(\text{example} = b) = \frac{L_b}{\sum_{u=0}^{B-1} L_u},
$$

so that, in expectation, each position in the dataset has equal chance to be chosen as an anchor.

To sample one key pair:

1. **Sample example index**

   $$
   b \sim \text{Categorical}\left( \frac{L_b}{\sum_u L_u} \right).
   $$

   If $ L_b < 2 $, resample (cannot form a pair).

2. **Sample anchor position**

   $$
   i \sim \text{Uniform}\{0, 1, \dots, L_b - 1\}.
   $$

3. **Compute local window boundaries**

   $$
   \text{left} = \max(0, i - \Delta_b), \quad
   \text{right} = \min(L_b - 1, i + \Delta_b).
   $$

4. **Define candidate neighbors**

   $$
   \mathcal{N}(i) = \{ j \in \{\text{left}, \dots, \text{right}\} : j \neq i \}.
   $$

   If $ \mathcal{N}(i) $ is empty (pathological small case), resample.

5. **Sample neighbor position**

   $$
   j \sim \text{Uniform}(\mathcal{N}(i)).
   $$

6. **Randomize ordering**  
   To avoid directional bias:

   - With probability $ 1/2 $, set $ (k_1, k_2) = (i, j) $.
   - With probability $ 1/2 $, set $ (k_1, k_2) = (j, i) $.

7. **Return pair**  
   The sampled key pair is represented by $ (b, k_1, k_2) $.

Repeat steps 1–7 until $ N $ pairs have been collected.

---

## 5. Properties of the Procedure

1. **Uniform coverage**  
   - Examples are drawn with probability $ L_b / \sum_u L_u $.
   - Within an example, anchor positions are uniform on $ \{0, \dots, L_b - 1\} $.  

   This makes anchors approximately uniformly distributed over all positions in the dataset.

2. **Locality**  
   Neighbor positions are constrained to be within $ \Delta_b \approx 0.05 L_b $ of the anchor, so the distance between keys in a pair satisfies:

   $$
   |k_1 - k_2| \le \Delta_b.
   $$

3. **Same-example pairs**  
   Both keys in a pair come from the same example index $ b $, by construction.

4. **No self-pairs**  
   We explicitly enforce $ j \neq i $, so $ k_1 \neq k_2 $.

5. **Symmetry in ordering**  
   Randomizing the order of $ (i, j) $ into $ (k_1, k_2) $ ensures the model is not systematically evaluated on “earlier vs later” in a fixed direction.

---

## 6. Numpy-style Implementation Sketch

Below is a reference implementation that returns an array of `(example_id, pos1, pos2)`:

```python
import numpy as np

def sample_key_pairs(lengths, num_pairs, max_rel_dist=0.05, rng=None):
    """
    Sample key pairs for attention plasticity estimation.

    Args:
        lengths: 1D array of shape (B,) with sequence lengths L_b per example.
        num_pairs: number of (example, pos1, pos2) pairs to sample.
        max_rel_dist: maximum distance between keys as a fraction of sequence length
                      (e.g. 0.05 for 5%).
        rng: optional numpy.random.Generator.

    Returns:
        pairs: int array of shape (num_pairs, 3), where each row is (b, k1_pos, k2_pos).
    """
    if rng is None:
        rng = np.random.default_rng()

    lengths = np.asarray(lengths)
    B = len(lengths)

    # Example sampling probabilities proportional to sequence length
    total_positions = lengths.sum()
    ex_probs = lengths / total_positions

    pairs = []

    while len(pairs) < num_pairs:
        # 1) Sample example
        b = rng.choice(B, p=ex_probs)
        L = lengths[b]
        if L < 2:
            continue  # cannot form a pair

        # 2) Distance threshold
        delta = max(1, int(np.floor(max_rel_dist * L)))

        # 3) Sample anchor position
        i = rng.integers(L)

        # 4) Neighbor window
        left = max(0, i - delta)
        right = min(L - 1, i + delta)

        # 5) Candidate neighbors excluding self
        neighbors = np.arange(left, right + 1)
        if neighbors.size <= 1:
            continue  # no valid neighbor
        neighbors = neighbors[neighbors != i]

        # 6) Sample neighbor
        j = rng.choice(neighbors)

        # 7) Randomize order
        if rng.random() < 0.5:
            k1, k2 = i, j
        else:
            k1, k2 = j, i

        pairs.append((b, k1, k2))

    return np.array(pairs, dtype=int)
```

---

## 7. Usage for Head-level Attention Plasticity

Given:

* A time-dependent query distribution $ q_t \sim \mathcal{N}(\mu_t, \Sigma_t) $,
* A set of sampled pairs $ (b, k_1, k_2) $ at time $ t $,

you can:

1. Compute $ p_t $ for each pair:

   $$
   p_t(k_1, k_2) = \mathbb{P}\big((q_t, k_1) > (q_t, k_2)\big),
   $$

   derived from the projected 1D Gaussian of $ (k_1 - k_2)^\top q_t $.

2. Compute pairwise plasticity:

   $$
   \mathrm{AP}_t(k_1, k_2) = 4p_t(k_1, k_2)\big(1 - p_t(k_1, k_2)\big).
   $$

3. Average over sampled pairs to obtain the head-level attention plasticity at time $ t $:

   $$
   \mathrm{AP}*t^{\text{head}} = \frac{1}{N} \sum*{n=1}^{N} \mathrm{AP}_t(k_1^{(n)}, k_2^{(n)}).
   $$

Because the sampling is uniform over positions and constrained to local neighborhoods, $ \mathrm{AP}_t^{\text{head}} $ captures genuine sensitivity to local key preferences rather than artifacts from global positional biases.
