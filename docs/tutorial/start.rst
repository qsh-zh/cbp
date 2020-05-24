Quick Start Guide
=================

Here we do run a simple constrained belief propagation on HMM.


Build the graph from hmm_builder
--------------------------------
::

    graph = cbp.graph.HMMBuilder(length=3,node_dim=3,policy=cbp.graph.coef_policy.avg_policy)

Run Belief Propagation
----------------------
::

    graph.run_cnp() # default is constrained norm-product
    graph.run_bp() # run iterative scaling belief propagation

Access to the marginal
----------------------
::

    marginal_1 = graph.get_node(f"VarNode_{1:03d}")
