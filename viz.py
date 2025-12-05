import struct

ops = [
    "unset", "var", "const", "+", "-", "*", "/", 
    "neg", "pow", "sqrt", "fabs", "cbrt", "log", "exp", 
    "sin", "cos", "tan", "asin", "acos", "atan", 
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", 
    "<", "<=", ">", ">=", 
    "not", "if", "and", "or"
]

def parse_input(text):
    nodes = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in line.split(",") if p != ""]
        nums = list(map(int, parts))
        if len(nums) < 3:
            continue
        node_id, class_id, opcode = nums[:3]
        args = nums[3:]
        nodes[node_id] = {"class": class_id, "opcode": opcode, "args": args}
    return nodes

def decode_constant(i):
    return struct.unpack(">f", struct.pack(">I", i))[0]

def generate_dot(nodes):
    dot = []
    dot.append("digraph G {")
    dot.append('  compound=true;')
    dot.append("  rankdir=LR;")
    dot.append('  node [fontsize=12];')
    dot.append('  edge [fontsize=10];')

    # Collect classes
    classes = {}
    for nid, n in nodes.items():
        classes.setdefault(n["class"], []).append(nid)

    # Emit clusters
    for cls_id, node_ids in sorted(classes.items()):
        dot.append(f'  subgraph cluster_class{cls_id} {{')
        dot.append(f'    label="Class {cls_id}";')
        dot.append('    style="rounded";')
        for nid in sorted(node_ids):
            node = nodes[nid]
            op = node["opcode"]
            label = f'Node {nid}'
            # Add var/const info inside node
            if op == 1 and node["args"]:
                label += f'\\nVar v{node["args"][0]}'
            elif op == 2 and node["args"]:
                fval = decode_constant(node["args"][0])
                label += f'\\nConst {fval:g}'
            else:
                label += f'\\n{ops[op]}'

            dot.append(f'    node{nid} [label="{label}", shape=circle];')
        dot.append("  }")

    # Edge styles by arg index
    styles = {0: "solid", 1: "dashed", 2: "dotted"}

    # Emit edges
    for nid, node in sorted(nodes.items()):
        opcode = node["opcode"]
        for a_idx, a_val in enumerate(node["args"]):
            if opcode >= 3:
                # Arg points to a class: connect to one node in that class
                target_cls = a_val
                if target_cls in classes:
                    target_node = classes[target_cls][0]  # pick first node in class
                    st = styles.get(a_idx,1)
                    dot.append(f'  node{nid} -> node{target_node} [style={st}, lhead=cluster_class{target_cls}];')
            # For opcode 1 and 2, info already inside node, no edge needed

    dot.append("}")
    return "\n".join(dot)

if __name__ == "__main__":
    input_text = """0,1,1,0,
1,2,2,1065353216,
2,1,8,1,2,
3,4,3,1,3,
4,5,9,4,
5,4,3,3,1,
6,5,8,5,2,
7,2,8,2,2,
8,1,8,3,2,
9,4,8,4,2,"""

    nodes = parse_input(input_text)
    dot = generate_dot(nodes)
    print(dot)