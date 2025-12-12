# Attention Plasticity with Full Query Covariance

This document describes how attention plasticity is computed when the query
distribution is modeled as a **full-covariance** multivariate normal that
varies with time.

We start from a time-varying model of the query vector and derive the
pairwise and head-level attention plasticity using the full covariance
matrix.

---

## 1. Query model

At each time step $t$, the query vector $q_t \in \mathbb{R}^d$ is modeled as

$$
q_t \sim \mathcal{N}(\mu_t, \Sigma_t),
$$

where

- $\mu_t \in \mathbb{R}^d$ is the mean query at time $t$,
- $\Sigma_t \in \mathbb{R}^{d \times d}$ is the **full covariance matrix** at time $t$.

---

## 2. Event definition for a key pair

Given a pair of keys $(k_1, k_2) \in \mathbb{R}^d \times \mathbb{R}^d$, we define
the event that key $k_1$ receives a higher dot-product score than key $k_2$
at time $t$ as

$$
A_t = \{ (q_t, k_1) > (q_t, k_2) \},
$$

where $(\cdot, \cdot)$ denotes the standard dot product.

We are interested in the probability

$$
p_t(k_1, k_2) := \mathbb{P}(A_t) = \mathbb{P}\big( (q_t, k_1) > (q_t, k_2) \big).
$$

---

## 3. Reduction to a 1D Gaussian using full covariance

Define the key difference

$$
w = k_1 - k_2 \in \mathbb{R}^d.
$$

Consider the scalar random variable

$$
Z_t = w^\top q_t.
$$

Because $q_t$ is multivariate normal with mean $\mu_t$ and covariance
$\Sigma_t$, the scalar $Z_t$ is 1D normal with

$$
Z_t \sim \mathcal{N}(m_t, v_t),
$$

where

$$
m_t = \mathbb{E}[Z_t] = w^\top \mu_t,
$$

$$
v_t = \mathrm{Var}(Z_t) = w^\top \Sigma_t\, w.
$$

The event $A_t$ can be written as

$$
A_t = \{ (q_t, k_1) > (q_t, k_2) \}
    = \{ w^\top q_t > 0 \}
    = \{ Z_t > 0 \}.
$$

Therefore

$$
p_t(k_1, k_2) = \mathbb{P}(A_t) = \mathbb{P}(Z_t > 0).
$$

If $v_t > 0$, then

$$
p_t(k_1, k_2)
= \Phi\!\left( \frac{m_t}{\sqrt{v_t}} \right),
$$

where $\Phi$ is the standard normal CDF.

If $v_t = 0$ (degenerate case), $Z_t$ is deterministic and we define

$$
p_t(k_1, k_2) =
\begin{cases}
1, & m_t > 0, \\
0, & m_t < 0, \\
\frac{1}{2}, & m_t = 0.
\end{cases}
$$

---

## 4. Attention plasticity for a single key pair

Attention plasticity for the pair $(k_1, k_2)$ at time $t$ is defined as

$$
\mathrm{AP}_t(k_1, k_2)
= 4\,\mathrm{Var}(\mathrm{Ber}(A_t))
= 4\,p_t(k_1, k_2)\bigl(1 - p_t(k_1, k_2)\bigr).
$$

Using the full-covariance-based expression for $p_t(k_1, k_2)$, we obtain

$$
\mathrm{AP}_t(k_1, k_2)
= 4\,\Phi\!\left(
  \frac{w^\top \mu_t}{\sqrt{w^\top \Sigma_t w}}
\right)
\left[
  1 - \Phi\!\left(
    \frac{w^\top \mu_t}{\sqrt{w^\top \Sigma_t w}}
  \right)
\right],
$$

for $v_t = w^\top \Sigma_t w > 0$, with the degenerate handling above if
$v_t = 0$.

---

## 5. Head-level attention plasticity

Let $\{(k_1^{(n)}, k_2^{(n)})\}_{n=1}^N$ be a set of sampled key pairs at time $t$.
For each pair, define

$$
w^{(n)} = k_1^{(n)} - k_2^{(n)},
$$

$$
m_t^{(n)} = (w^{(n)})^\top \mu_t,
\quad
v_t^{(n)} = (w^{(n)})^\top \Sigma_t\, w^{(n)},
$$

$$
p_t^{(n)} = 
\begin{cases}
\Phi\!\left( \dfrac{m_t^{(n)}}{\sqrt{v_t^{(n)}}} \right), & v_t^{(n)} > 0, \\
1, & v_t^{(n)} = 0,\ m_t^{(n)} > 0, \\
0, & v_t^{(n)} = 0,\ m_t^{(n)} < 0, \\
\frac{1}{2}, & v_t^{(n)} = 0,\ m_t^{(n)} = 0,
\end{cases}
$$

and

$$
\mathrm{AP}_t^{(n)} = 4\,p_t^{(n)}\bigl(1 - p_t^{(n)}\bigr).
$$

The **head-level attention plasticity** at time $t$ is then the average over
all sampled pairs:

$$
\mathrm{AP}_t^{\text{head}}
= \frac{1}{N} \sum_{n=1}^N \mathrm{AP}_t^{(n)}
= \frac{1}{N} \sum_{n=1}^N 4\,p_t^{(n)}\bigl(1 - p_t^{(n)}\bigr).
$$

---

## 6. Efficient batched computation

For batched implementation, stack the key differences into a matrix

$$
W \in \mathbb{R}^{N \times d}, \quad W_n = (w^{(n)})^\top.
$$

Then:

- All means at time $t$:

  $$
  m_t = W \mu_t \in \mathbb{R}^N.
  $$

- All variances at time $t$:

  $$
  v_t = \operatorname{diag}\bigl(W \Sigma_t W^\top\bigr)
      \in \mathbb{R}^N,
  $$

  i.e. the diagonal of the $N \times N$ matrix $W \Sigma_t W^\top$.

Compute elementwise

$$
z_t = m_t \oslash \sqrt{v_t},
$$

(where $\oslash$ denotes elementwise division, with appropriate handling of
zero variances), and then

$$
p_t = \Phi(z_t),
$$

$$
\mathrm{AP}_t^{(n)} = 4\,p_t^{(n)}\bigl(1 - p_t^{(n)}\bigr),
$$

before averaging them to obtain $\mathrm{AP}_t^{\text{head}}$.
