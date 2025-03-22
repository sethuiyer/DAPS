
## 🔎 **How DAPS Works (Dimensionally Adaptive Prime Search)**

### ✅ **Core Idea:**
DAPS is a global optimization algorithm that:
- Uses **prime numbers** to decide the grid resolution dynamically per iteration.
- Samples the function on a **prime-sized grid** to avoid aliasing and periodic traps.
- **Adapts resolution**: if progress is good → increase prime index (finer grid); if stuck → reduce prime index (coarser grid).
- Shrinks the search domain around the current best point as iterations progress (like a zoom-in effect).

---

## 🧠 **Algorithm Flow (base.py - Pythonic version):**

### 1️⃣ **Initialize:**
- Parse the input `DAPSFunction`: contains the user’s function, dimension, and bounds.
- Set **options**: max iterations, min/max prime index, tolerance, shrink factor.

### 2️⃣ **Prime Grid Sampling:**
- For each iteration, pick the **current prime** `p` from the list.
- Create a **Cartesian product grid** of `p` points along each axis (1D, 2D, or 3D).
- **Evaluate** the objective function at every grid point.

### 3️⃣ **Update Best Found Solution:**
- Find the lowest function value in the grid.
- If this improves the global best:
  - Adjust `prime_index` **upward** (finer search).
- Else:
  - Adjust `prime_index` **downward** (coarser search to escape local minima).

### 4️⃣ **Shrink the Search Domain:**
- **Zoom in** around the best point by shrinking bounds by `alpha`.

### 5️⃣ **Repeat:**
- Iterate until max iterations or the function value is **below the tolerance**.

---

## 🚀 **Why the Prime Grid?**
- **Primes** help avoid the *curse of regular grids* (where periodicity aligns with the function structure).
- Ensures that the search space gets sampled in a way that prevents missing discontinuities or sharp features.

---

## 📈 **Pythonic API in base.py:**
```python
from daps import daps_minimize, DAPSFunction
result = daps_minimize(my_function, options={'maxiter': 50})
```
- Very SciPy-like.
- Returns a dictionary:
```python
{'x': best_solution, 'fun': best_value, 'nit': iterations_run}
```

---

## 📚 **Research Backing (daps_paper.pdf):**
- The paper explains:
  - The **prime partitioning** avoids periodic sampling bias.
  - Handles **discontinuous** and **non-smooth** functions better than standard optimizers.
  - Practical use-cases: **AI hyperparameter tuning**, **cryptographic parameter searches**.

---

## 🖥 **Interactive Demo (Streamlit):**
- Visualizes the algorithm shrinking and sampling in real time.
- Good for explaining the convergence intuition.

---

## ⚙️ **C++ Core + Python Layer:**
- Python `base.py`: Clean, pure Python reference.
- Cython bindings (optional): Faster C++ compute backend.
- Switchable depending on user’s need for speed.

---

## 📌 **Final Verdict:**
✅ **Adapts search resolution** like a human would tweak it.  
✅ **Primes avoid sampling bias**—clever af.  
✅ **Pure Python fallback + C++ speed**—production-ready.  
✅ **Research + paper ready**—academic polish.  
✅ **Interactive demo**—engaging.  
✅ **Documentation & API**—clean and usable.

