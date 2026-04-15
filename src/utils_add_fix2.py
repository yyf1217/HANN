from . import *
from torch_geometric.utils import k_hop_subgraph
import networkx as nx
from collections import defaultdict

class Subgraph(data.Data):
    def __inc__(self, key, *args, **kwargs):
        if key in ("index_u", "index_v"): return self.num_node
        elif "index" in key: return self.num_nodes
        else: return 0


def subgraph(graph):
    
    # print("subgraph called")
    num_hops = 1
    max_dist = 2
    
    node = torch.arange((N:=graph.num_nodes) ** 2).view(size=(N, N))
    adj = pyg.utils.to_dense_adj(graph.edge_index, max_num_nodes=N).squeeze(0)

    spd = torch.where(~torch.eye(N, dtype=bool) & (adj == 0), torch.full_like(adj, float("inf")), adj)
    for k in range(N): spd = torch.minimum(spd, spd[:, [k]] + spd[[k], :])

    # print('spd')
    # print(spd)
    # print('spd.shape')
    # print(spd.shape)

    # ===== 4. 根据 spd <= max_dist 选节点对 =====
    pair_mask = (spd <= max_dist)

    pair_mask = torch.triu(pair_mask, diagonal=0)  # 保留 u <= v

    pair_index = pair_mask.nonzero(as_tuple=False)  # [num_pairs, 2]

    # print('pair_index')
    # print(pair_index)
    # print('graph.edge_index')
    # print(graph.edge_index)
    # homotopy = torch.zeros(spd.shape, device=graph.edge_index.device)
    homotopy = torch.zeros((*spd.shape, 5), device=graph.edge_index.device)
    homology = torch.zeros((*spd.shape, 1), device=graph.edge_index.device)

    # t = 0
    for pair_id, (u, v) in enumerate(pair_index.tolist()):
        seed_nodes = torch.tensor([u, v], dtype=torch.long, device=graph.edge_index.device)

        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=seed_nodes,
            num_hops=num_hops,
            edge_index=graph.edge_index,
            relabel_nodes=True,
            num_nodes=graph.num_nodes
        )                 # 有些 graph 存在孤立节点
        # print("graph.num_nodes =", graph.num_nodes)
        # print("edge_index.max() =", int(graph.edge_index.max()))
        # print("edge_index.min() =", int(graph.edge_index.min()))
        h_uv = com_homotopy(u, v, subset, sub_edge_index)
        hl_uv = compute_betti_number(u, v, subset, sub_edge_index)
        # homotopy[u, v] = h_uv
        homotopy[u, v, :] = h_uv
        homology[u, v, :] = hl_uv

        # print('pair_id')
        # print(pair_id)
        # print('(u, v)')
        # print((u, v))
        # print('subset')
        # print(subset)
        # print('sub_edge_index')
        # print(sub_edge_index)
        # print('mapping')
        # print(mapping)
        # print('edge_mask')        
        # print(edge_mask)
        # print('hl_uv')        
        # print(hl_uv)
        # # print(debug)
        # # print(t)
        # t += 1 
        # if t == 100:
        #     print(debug)

    # print('homotopy')
    # print(homotopy)
    # print(debug)

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
        h=homotopy.to(torch.float32).flatten(end_dim=1),
        hl=homology.to(torch.float32).flatten(end_dim=1),

        index_d=node[:, 0] + node[0, :],

        index_u=torch.broadcast_to(node[0, :, None], (N, N)).flatten(),
        index_v=torch.broadcast_to(node[0, None, :], (N, N)).flatten(),

        index_uL=index_uL,
        index_vL=index_vL,
        edge_index=edge_index,

        index_uLF=stack(node[:, 0] + dst[:, None], node[:, 0] + src[:, None], (N * src + dst)[:, None]),
        index_vLF=stack(node[0] + N * dst[:, None], node[0] + N * src[:, None], (N * dst + src)[:, None]),
    )



def com_homotopy(u, v, subset, sub_edge_index, max_len=5):
    """
    参数
    ----
    u, v : int
        原图中的节点编号
    subset : 1D Tensor / list
        subgraph 所有节点在原图中的编号，subset[i] 是局部节点 i 对应的原图编号
    sub_edge_index : Tensor, shape [2, m]
        subgraph 的边（局部编号）
    max_len : int
        枚举 u -> v 的所有长度 <= max_len 的路径（允许节点重复）

    返回
    ----
    homotopy : dict
        key = 缩减后路径长度
        value = {
            "total_paths": 该长度下缩减后的总路径数（去重前）,
            "unique_paths": 该长度下不重复路径数,
            "classes": 每个等价类中的路径（原图编号）
        }

    num_homotopy : Tensor, shape [max_len+1]
        num_homotopy[i] 表示“缩减后长度为 i”的不重复路径数量
    """

    # ===== 1. 预处理 =====
    if isinstance(subset, torch.Tensor):
        subset_list = subset.tolist()
    else:
        subset_list = list(subset)

    if not isinstance(sub_edge_index, torch.Tensor):
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long)

    n = len(subset_list)

    orig2local = {orig_id: local_id for local_id, orig_id in enumerate(subset_list)}
    if u not in orig2local or v not in orig2local:
        raise ValueError(f"u={u} 或 v={v} 不在 subset 中。")

    u_local = orig2local[u]
    v_local = orig2local[v]

    # 无向图邻接表
    adj = [set() for _ in range(n)]
    for a, b in sub_edge_index.t().tolist():
        adj[a].add(b)
        adj[b].add(a)

    def has_edge(a, b):
        return b in adj[a]

    # ===== 2. 路径缩减 =====
    def reduce_path(path):
        """
        对路径反复做缩减，直到不能再缩减。

        缩减规则（按 1step -> 2step -> 3step 的顺序反复执行）：

        1step:
            若出现 a-b-c-d，且 a-d 有边，则 a-b-c-d -> a-d
            （窗口大小 4，缩成 2，允许包含 u 和 v）

        2step:
            若出现 a-b-c，且 a-c 有边，则 a-b-c -> a-c
            （窗口大小 3，缩成 2，允许包含 u 和 v）

        3step:
            若出现 a-b-a，则 a-b-a -> a
            （允许包含 u 和 v）

        4step:
            若最终缩减为 [x] 或 [x, x]，则消去该路径，返回 None
        """
        path = path.copy()

        while True:
            changed = False

            # -------- 1step: a-b-c-d -> a-d, if edge(a, d) --------
            i = 0
            while i + 3 < len(path):
                a, b, c, d = path[i:i + 4]
                if has_edge(a, d):
                    path = path[:i] + [a, d] + path[i + 4:]
                    changed = True
                    break
                i += 1
            if changed:
                continue

            # -------- 2step: a-b-c -> a-c, if edge(a, c) --------
            i = 0
            while i + 2 < len(path):
                a, b, c = path[i:i + 3]
                if has_edge(a, c):
                    path = path[:i] + [a, c] + path[i + 3:]
                    changed = True
                    break
                i += 1
            if changed:
                continue

            # -------- 3step: a-b-a -> a --------
            i = 0
            while i + 2 < len(path):
                a, b, c = path[i:i + 3]
                if a == c:
                    path = path[:i] + [a] + path[i + 3:]
                    changed = True
                    break
                i += 1
            if changed:
                continue

            # 没有变化，停止
            break

        # -------- 4step: 消去退化路径 --------
        if len(path) == 1:
            return None
        if len(path) == 2 and path[0] == path[1]:
            return None

        return path

    # ===== 3. 枚举长度 <= max_len 的所有 walk，并做缩减 =====
    reduced_paths_by_len = defaultdict(list)

    def dfs(cur, depth, path):
        """
        depth: 已走边数
        path : 当前局部路径
        """
        if depth > max_len:
            return

        if cur == v_local:
            reduced = reduce_path(path)
            if reduced is not None:
                reduced_len = len(reduced) - 1
                if 0 <= reduced_len <= max_len:
                    reduced_paths_by_len[reduced_len].append(reduced.copy())

        if depth == max_len:
            return

        for nxt in adj[cur]:
            path.append(nxt)
            dfs(nxt, depth + 1, path)
            path.pop()

    dfs(u_local, 0, [u_local])

    # ===== 4. “重复路径”判定 =====
    # def is_duplicate_path(path1, path2):
    #     """
    #     两条同长度路径，若在每个中间位置对应节点：
    #     - 相同，或
    #     - 两者直接有边
    #     则认为重复
    #     """
    #     if len(path1) != len(path2):
    #         return False
    #     if path1[0] != path2[0] or path1[-1] != path2[-1]:
    #         return False

    #     for x, y in zip(path1[1:-1], path2[1:-1]):
    #         if x == y:
    #             continue
    #         if not has_edge(x, y):
    #             return False
    #     return True
    def is_duplicate_path(path1, path2):
        """
        两条同长度路径，若满足以下任一条件，则认为重复：

        规则1：
        在每个中间位置对应节点：
        - 相同，或
        - 两者直接有边

        规则2：
        找到两条路径中“对应位置上节点相同”的那些位置，
        如果这些位置相邻之间的下标距离都 <= 2，则认为重复
        """
        if len(path1) != len(path2):
            return False
        if path1[0] != path2[0] or path1[-1] != path2[-1]:
            return False
        
        # 规则1：每个中间位置对应节点相同或直接相连
        all_middle_match = True
        for x, y in zip(path1[1:-1], path2[1:-1]):
            if x == y:
                continue
            if not has_edge(x, y):
                all_middle_match = False
                break
        if all_middle_match:
            return True
    
        # ===== 相同节点位置的距离判定（相邻位置版本） =====
        same_pos = []
        for idx, (x, y) in enumerate(zip(path1, path2)):
            if x == y:
                same_pos.append(idx)

        if len(same_pos) >= 2:
            ok = True
            for i in range(len(same_pos) - 1):
                if same_pos[i + 1] - same_pos[i] > 2:
                    ok = False
                    break
            if ok:
                # if len(same_pos) >= 3:
                    # print(path1)
                    # print(path2)
                    # print(debug)
                return True

    # ===== 5. 按缩减后长度做归类 =====
    homotopy = {}
    num_homotopy = torch.zeros(max_len, dtype=torch.long)

    for length in range(1, max_len + 1):
        paths = reduced_paths_by_len.get(length, [])
        k = len(paths)

        if k == 0:
            homotopy[length] = {
                "total_paths": 0,
                "unique_paths": 0,
                "classes": []
            }
            continue

        visited = [False] * k
        classes = []

        for i in range(k):
            if visited[i]:
                continue

            stack = [i]
            visited[i] = True
            comp = []

            while stack:
                x = stack.pop()
                comp.append(paths[x])

                for y in range(k):
                    if not visited[y] and is_duplicate_path(paths[x], paths[y]):
                        visited[y] = True
                        stack.append(y)

            classes.append(comp)

        # 转回原图编号
        classes_in_orig = []
        for comp in classes:
            comp_orig = []
            for path in comp:
                comp_orig.append([subset_list[node] for node in path])
            classes_in_orig.append(comp_orig)

        homotopy[length] = {
            "total_paths": k,
            "unique_paths": len(classes),
            "classes": classes_in_orig
        }
        num_homotopy[length-1] = len(classes)


    # print('homotopy')
    # print(homotopy)
    # print('num_homotopy')
    # print(num_homotopy)

    # return sum(num_homotopy)
    return num_homotopy


def _build_subgraph_inputs(start_node, end_node, subset, sub_edge_index):
    subset_list = [int(x) for x in _to_list(subset)]
    if start_node not in subset_list or end_node not in subset_list:
        raise ValueError("start_node or end_node is not in subset.")

    local_edges = _parse_edge_index(sub_edge_index)
    local_to_orig = {i: node for i, node in enumerate(subset_list)}

    edge_pairs = []
    for u_local, v_local in local_edges:
        if u_local not in local_to_orig or v_local not in local_to_orig:
            raise ValueError("sub_edge_index has invalid local index.")
        edge_pairs.append((local_to_orig[u_local], local_to_orig[v_local]))

    return subset_list, _to_undirected_edges(edge_pairs)

def compute_betti_number(
    start_node,
    end_node,
    subset=None,
    sub_edge_index=None,
    num_hops=1,
    graph=None,
    num_nodes=None,
    edge_index=None,
):
    """
    Two usage modes:
    1) Manual subgraph mode :
       compute_betti_number(start_node, end_node, subset, sub_edge_index)
    2) Automatic k-hop mode :
       compute_betti_number(start_node, end_node, num_nodes=N, edge_index=edge_index, num_hops=1)
       or
       compute_betti_number(start_node, end_node, graph=pyg_graph, num_hops=1)
    """

    subset_list, edges = _build_subgraph_inputs(start_node, end_node, subset, sub_edge_index)

    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(subset_list)
    graph_nx.add_edges_from(edges)

    n = graph_nx.number_of_nodes()
    m = graph_nx.number_of_edges()
    c = nx.number_connected_components(graph_nx)

    return m - n + c

def _to_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)

def _parse_edge_index(edge_index):
    raw = _to_list(edge_index)
    if len(raw) == 0:
        return []

    # 2 x E format
    if isinstance(raw[0], (list, tuple)) and len(raw) == 2 and (
        len(raw[0]) == 0 or not isinstance(raw[0][0], (list, tuple))
    ):
        return [(int(u), int(v)) for u, v in zip(raw[0], raw[1])]

    # E x 2 format
    if isinstance(raw[0], (list, tuple)):
        return [(int(u), int(v)) for u, v in raw]

    raise ValueError("edge_index format is not supported.")


def _to_undirected_edges(edge_pairs):
    edges = set()
    for u, v in edge_pairs:
        if u == v:
            continue
        edges.add((u, v) if u < v else (v, u))
    return list(edges)