# Modeling Time-Varying Query Distributions with an Online Estimator

This document explains how we model the distribution of query vectors $q_t$ over time using an online estimator with exponential forgetting. The queries are $d$-dimensional and are modeled as a multivariate normal distribution with a **time-varying mean** and a **full covariance matrix**.

---

## 1. Setup

At each discrete time step $t = 1, 2, \dots, T$, we consider a random query vector
$Q_t \in \mathbb{R}^d$.

We approximate its distribution as a Gaussian:
$$
Q_t \sim \mathcal{N}(\mu_t, \Sigma_t),
$$
where

- $\mu_t \in \mathbb{R}^d$ is the mean at time $t$,
- $\Sigma_t \in \mathbb{R}^{d \times d}$ is the **full covariance matrix** at time $t$.

In practice, we only observe query samples at some time steps. Let

- $x_t \in \mathbb{R}^d$ denote the observed query at time $t$ (if available),
- $\delta_t \in \{0, 1\}$ indicate whether we have an observation at time $t$:
  - $\delta_t = 1$ if $x_t$ is observed,
  - $\delta_t = 0$ otherwise.

Our goal is to maintain estimates $(\mu_t, \Sigma_t)$ for **every** time step $t$,
even when $\delta_t = 0$.

---

## 2. Exponentially Weighted Online Estimator

We use an **exponentially weighted moving average** (EWMA) scheme with a forgetting factor
$\alpha \in (0, 1]$.

Intuition:

- $\alpha$ controls **how quickly** the model adapts to new data.
- Larger $\alpha$ means higher "plasticity": recent queries dominate the estimates.
- Smaller $\alpha$ gives smoother, more stable estimates over time.

To estimate the full covariance, we track:

1. The mean $\mu_t$,
2. The second moment matrix $M_t \approx \mathbb{E}[Q_t Q_t^\top]$,
3. The covariance $\Sigma_t = M_t - \mu_t \mu_t^\top$.

### 2.1. Initialization

Choose initial values at $t = 0$, for example:

- $\mu_0 = 0 \in \mathbb{R}^d$ (or the first observed query),
- $M_0 = I_d$ or $M_0 = 0 \in \mathbb{R}^{d \times d}$,

then
$$
\Sigma_0 = M_0 - \mu_0 \mu_0^\top.
$$

These are design choices and can be tuned depending on the application.

---

### 2.2. Update when a query is observed

When $\delta_t = 1$, we observe a query sample $x_t \in \mathbb{R}^d$. We update:

1. **Mean:**
   $$
   \mu_t = (1 - \alpha)\,\mu_{t-1} + \alpha\,x_t.
   $$

2. **Second moment matrix:**
   $$
   M_t = (1 - \alpha)\,M_{t-1} + \alpha\,x_t x_t^\top.
   $$

3. **Covariance matrix:**
   $$
   \Sigma_t = M_t - \mu_t \mu_t^\top.
   $$

Notes:

- The outer product $x_t x_t^\top$ is a $d \times d$ matrix, so $\Sigma_t$ is a full covariance matrix.
- This scheme corresponds to an exponentially weighted estimator of
  $\mathbb{E}[Q_t]$ and $\mathbb{E}[Q_t Q_t^\top]$.

---

### 2.3. Update when no query is observed

When $\delta_t = 0$, there is no new sample at time $t$. In this case, we **carry
the estimates forward**:

$$
\mu_t = \mu_{t-1}, \quad
M_t = M_{t-1}, \quad
\Sigma_t = \Sigma_{t-1}.
$$

Conceptually, this is equivalent to not performing any update at that time step
(i.e. using an effective learning rate of zero at $t$).

As a result, we still have a well-defined Gaussian approximation
$Q_t \sim \mathcal{N}(\mu_t, \Sigma_t)$ for every time step $t$, whether or not
a query was observed at that exact time.

---

## 3. Interpretation

The above update rules implement an **exponentially decayed history** of queries:

- The effective weight of a sample $x_{t-k}$ in $\mu_t$ and $\Sigma_t$ decays roughly like $(1 - \alpha)^k$.
- Recent queries have more influence, and older queries are gradually forgotten.

Therefore:

- $\mu_t$ captures the **current typical direction** of the query vectors.
- $\Sigma_t$ captures their **current variability and correlations** across dimensions.

These parameters are then used as the time-varying Gaussian model:
$$
Q_t \sim \mathcal{N}(\mu_t, \Sigma_t),
$$
which can be plugged into downstream quantities, such as the probability
that one key receives higher attention than another and the resulting
attention plasticity measures.

---

## 4. Implementation Sketch (NumPy-style)

Below is a minimal reference implementation for estimating $\mu_t$ and $\Sigma_t$ over time:

```python
import numpy as np

def online_query_distribution(q_obs, observed_mask, alpha):
    """
    Online estimator for time-varying query distribution.

    Args:
        q_obs: array of shape (T, d), storing observed queries x_t.
               Values at times with no observation can be ignored.
        observed_mask: boolean array of shape (T,), True if x_t is observed.
        alpha: forgetting factor in (0, 1].

    Returns:
        mus: array of shape (T, d), estimated means mu_t.
        Sigmas: array of shape (T, d, d), estimated covariances Sigma_t.
    """
    T, d = q_obs.shape

    mus = np.zeros((T, d))
    Sigmas = np.zeros((T, d, d))

    # Initialization
    mu = np.zeros(d)
    M = np.zeros((d, d))

    for t in range(T):
        if observed_mask[t]:
            x = q_obs[t]

            # Update mean
            mu = (1 - alpha) * mu + alpha * x

            # Update second moment
            M = (1 - alpha) * M + alpha * np.outer(x, x)

        # Covariance from second moment and mean
        Sigma = M - np.outer(mu, mu)

        # (Optional: enforce symmetry and numerical stability)
        Sigma = 0.5 * (Sigma + Sigma.T)
        # Sigma += 1e-6 * np.eye(d)

        mus[t] = mu
        Sigmas[t] = Sigma

    return mus, Sigmas
```

This function returns the full time series ({\mu_t, \Sigma_t}_{t=1}^T), providing
a Gaussian model for the query distribution at each time step.

