import bito
import numpy as np
import time
import pandas as pd


SIMPLE_SPECIFICATION = bito.PhyloModelSpecification(
    substitution="JC69", site="constant", clock="none"
)


def time_gradients(iter_count, inst):
    start_time = time.time()
    for i in range(iter_count):
        _ = inst.phylo_gradients()
    return time.time() - start_time


tree_count = 10
thread_count = 10
iter_count = 3

inst = bito.unrooted_instance("flu")
inst.read_fasta_file("final.n.fasta")
inst.read_newick_file("final.n.fasta.treefile")
inst.load_duplicates_of_first_tree(tree_count)
inst.process_loaded_trees()
inst.print_status()
inst.prepare_for_phylo_likelihood(SIMPLE_SPECIFICATION, thread_count)

data = []

for iter_idx in range(iter_count):
    print(f"Starting iteration {iter_idx}.")
    for gradient_count in np.logspace(2, 3, num=4, dtype=int):
        print(f"Calculating {gradient_count} gradients.")
        result = time_gradients(gradient_count, inst)
        d = {
            "iter_idx": iter_idx,
            "tree_count": tree_count,
            "thread_count": thread_count,
            "gradient_count": gradient_count,
            "time": result,
        }
        data.append(d)

df = pd.DataFrame(data)

df.to_csv("timing_results.csv", index=False)
