def avg_policy(node, node_map):  # pylint: disable=unused-argument
    return 1.0 / len(node_map)


def factor_policy(node, node_map):
    cnt_var = 0
    cnt_factor = 0
    for name, _ in node_map.items():
        if 'VarNode' in name:
            cnt_var += 1
        elif 'FactorNode' in name:
            cnt_factor += 1
        else:
            raise RuntimeError(f"{name} is wrong")

    if node.__class__.__name__ == 'VarNode':
        coef = 0
    elif node.__class__.__name__ == "FactorNode":
        coef = 1.0 / cnt_factor
    return coef


def bp_policy(node, node_map):  # pylint: disable=unused-argument
    if node.__class__.__name__ == 'VarNode':
        coef = 1.0 - len(node.connections)
    elif node.__class__.__name__ == 'FactorNode':
        coef = 1.0
    return coef


def crazy_policy(node, node_map):
    cnt_var = 0
    cnt_factor = 0
    propotion_coef = 2.0
    for name, _ in node_map.items():
        if 'VarNode' in name:
            cnt_var += 1
        elif 'FactorNode' in name:
            cnt_factor += 1
        else:
            raise RuntimeError(f"{name} is wrong")

    if node.__class__.__name__ == 'VarNode':
        coef = 1.0 - len(node.connections) - propotion_coef * \
            (cnt_factor + cnt_var) / cnt_var
    elif node.__class__.__name__ == "FactorNode":
        coef = 1 + propotion_coef * (cnt_factor + cnt_var) / cnt_factor
    return coef
