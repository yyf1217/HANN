from . import *

class Subgraph(data.Data):
    def __inc__(self, key, *args, **kwargs):
        if key in ("index_u", "index_v"): return self.num_node
        elif "index" in key: return self.num_nodes
        else: return 0


def subgraph(graph):
    
    # print("subgraph called")
    
    node = torch.arange((N:=graph.num_nodes) ** 2).view(size=(N, N))
    adj = pyg.utils.to_dense_adj(graph.edge_index, max_num_nodes=N).squeeze(0)

    spd = torch.where(~torch.eye(N, dtype=bool) & (adj == 0), torch.full_like(adj, float("inf")), adj)
    for k in range(N): spd = torch.minimum(spd, spd[:, [k]] + spd[[k], :])

    attr, (dst, src) = graph.edge_attr, graph.edge_index
    if attr is not None and attr.ndim == 1: attr = attr[:, None]
    assert graph.x.ndim == 2
    
    stack = lambda *x: torch.stack(torch.broadcast_tensors(*x)).flatten(start_dim=1)

    index_uL = stack(node[:, 0] + dst[:, None], node[:, 0] + src[:, None])  # [2, N*E]
    index_vL = stack(node[0] + N * dst[:, None], node[0] + N * src[:, None])
    
    edge_index = torch.cat([index_uL, index_vL], dim=1).long()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index, _ = pyg.utils.coalesce(edge_index, None, num_nodes=N * N)
    
    return Subgraph(
        num_node=N,
        num_nodes=N**2,
        x=graph.x[None].repeat_interleave(N, dim=0).flatten(end_dim=1),
        y=graph.y,
        a=attr[:, None].repeat_interleave(N, dim=1).flatten(end_dim=1) if attr is not None else None,

        e=adj.to(int).flatten(end_dim=1),
        d=spd.to(int).flatten(end_dim=1),

        index_d=node[:, 0] + node[0, :],

        index_u=torch.broadcast_to(node[0, :, None], (N, N)).flatten(),
        index_v=torch.broadcast_to(node[0, None, :], (N, N)).flatten(),

        index_uL=index_uL,
        index_vL=index_vL,
        edge_index=edge_index,

        index_uLF=stack(node[:, 0] + dst[:, None], node[:, 0] + src[:, None], (N * src + dst)[:, None]),
        index_vLF=stack(node[0] + N * dst[:, None], node[0] + N * src[:, None], (N * dst + src)[:, None]),
    )
