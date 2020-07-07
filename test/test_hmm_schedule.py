# %%
%matplotlib inline
%load_ext autoreload
%autoreload 2

# %% [markdown]
## Test effect of Schedule

# %% [markdown]
### Original performance

# %%
import sys
sys.path.insert(0, '..')
# %%
import numpy as np
from cbp.builder import HMMBuilder
from cbp.graph.coef_policy import bp_policy
# %%
def one_direction():
    rng = np.random.RandomState(1)
    for i in range(10):
        num_node = int(rng.randint(10, 15))
        node_dim = int(rng.randint(4, 6))
        graph = HMMBuilder(num_node, node_dim, bp_policy)()
        graph.run_cnp()

# %%
%timeit -r3 -n3 one_direction()

# %%
