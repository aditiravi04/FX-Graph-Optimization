import torch
import torch.fx
from fuse_operators import FuseFunc


def fuse_linear_relu(model: torch.nn.Module):


    # get full computation graph w/ FX as we did in graphTracer
    traced = torch.fx.symbolic_trace(model)
    graph = traced.graph

    # dict for name module look up (to check if call_module node is linear/ReLU)
    modules = dict(traced.named_modules())

    # look for linear, ReLU pattern in graph , check each node
    for node in list(graph.nodes):

        # only care about call_modules for now since we have nn.Sequential
        if node.op != "call_module":
            continue
        
        # check module type
        if not isinstance(modules[node.target], torch.nn.Linear):
            continue

        # safety check to see if only one node takes this linear node's output (would break if multiple take in)
        if len(node.users) != 1:
            continue
        
        # check next node
        next_node = node.next

        # later can check if ReLU is call_module or call_function (i.e. someone calls F.relu(x) directly in fwd())
        # if next node is ReLU call_module
        if (
            next_node
            and next_node.op == "call_module"
            and isinstance(modules[next_node.target], torch.nn.ReLU)
        ):
            # we are fusing! :D
            print(f"Fusing {node.name} + {next_node.name}")

            # call FuseFunc to create fused module to repalce
            fused_module = FuseFunc(modules[node.target])
            fused_name = f"{node.name}_relu_fused"

            # register in traced
            setattr(traced, fused_name, fused_module)

            # insert new node in graph that calls fused_module using same input as og linear module
            with graph.inserting_after(node):
                new_node = graph.call_module(
                    fused_name,
                    args=node.args,
                )
            # anything that used the ReLU module output should now point to our fused module
            next_node.replace_all_uses_with(new_node)

            # remove old linear, relu nodes; 
            graph.erase_node(next_node) #erase relu first b/c linear depends on it
            graph.erase_node(node)

    # cechk graph still valid
    graph.lint()
    traced.recompile()

    return traced