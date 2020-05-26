from cbp.utils.np_utils import reduction_ndarray


def get_node2root(node):
    rtn = []
    tmp = node
    while True:
        rtn.append(tmp)
        tmp = tmp.parent
        if not tmp:
            break

    return rtn


def find_link(node_a, node_b):
    """find the path link from node_a to node_b in a tree graph, following:

    * Find paths from node to root seperately
    * merge two path and erase the repeated node

    :param node_a: start node
    :type node_a: BaseNode
    :param node_b: tail node
    :type node_b: BaseNode
    :return: a list of node
    :rtype: list
    """
    a_2root = get_node2root(node_a)
    b_2root = get_node2root(node_b)
    while (len(b_2root) > 1 and len(a_2root) > 1):
        if b_2root[-1] == a_2root[-1] and b_2root[-2] == a_2root[-2]:
            b_2root.pop()
            a_2root.pop()
        else:
            break
    b_2root.reverse()
    if len(b_2root) >= 2:
        return a_2root + b_2root[1:]

    return a_2root[:-1] + b_2root[:]


def itsbp_inner_loop(loop_link):
    if len(loop_link) == 2:
        return

    for sender, receiver in zip(loop_link[0:-1], loop_link[1:]):
        sender.send_message(receiver)

    loop_link.reverse()
    for sender, receiver in zip(loop_link[0:-1], loop_link[1:]):
        sender.send_message(receiver)


def cal_marginal_from_tensor(prob_tensor, varnode_list):
    """calculate the marginal from a joint distribution tensor

    :param prob_tensor: joint tensor
    :type prob_tensor: ndarray
    :param varnode_list: which dims to calculate the marginal
    :type varnode_list: list
    :return: marginal
    :rtype: list
    """
    rtn_marginal = []
    for i, _ in enumerate(varnode_list):
        rtn_marginal.append(reduction_ndarray(prob_tensor, i))
    return rtn_marginal
