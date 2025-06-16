import re
import json
import numpy as np
import scipy.sparse as sp
import os
import random
from collections import defaultdict

def parse_single_circuit(verilog_filepath, trojan_filepath, design_name):
    """
    Parses a single Verilog file and returns its graph components with prefixed node names.
    (This helper function remains the same as before).
    """
    with open(verilog_filepath, 'r') as f:
        verilog_content = f.read()

    trojan_local_gates = set()
    if os.path.exists(trojan_filepath):
        with open(trojan_filepath, 'r') as f:
            trojan_content = f.read()
        in_trojan_section = False
        for line in trojan_content.strip().split('\n'):
            if line == 'END_TROJAN_GATES': in_trojan_section = False
            if in_trojan_section: trojan_local_gates.add(line.strip())
            if line == 'TROJAN_GATES': in_trojan_section = True

    module_content_match = re.search(r'module.*?;(.*?)endmodule', verilog_content, re.DOTALL)
    if not module_content_match: return None
    module_content = module_content_match.group(1)

    primary_inputs = {pi.strip() for match in re.findall(r'input\s+(?:\[\d+:\d+\])?\s*([\w\s,]+);', module_content) for pi in match.split(',')}
    gate_insts = re.compile(r'(\w+)\s+([a-zA-Z_]\w*)\s*\((.*?)\);', re.MULTILINE).findall(module_content)

    wire_to_driver, node_names, node_types = {}, set(), {}
    
    # 1st pass: Find drivers and node names
    for gate_type, gate_name, ports_str in gate_insts:
        gate_name_prefixed = f"{design_name}_{gate_name.strip()}"
        node_names.add(gate_name_prefixed)
        node_types[gate_name_prefixed] = gate_type.strip()
        output_wire = ports_str.split(',')[0].strip()
        named_port_match = re.match(r'\.\w+\s*\((.*?)\)', output_wire)
        if named_port_match: output_wire = named_port_match.group(1).strip()
        wire_to_driver[output_wire] = gate_name_prefixed
        
    for pi in primary_inputs:
        pi_name_prefixed = f"{design_name}_{pi}"
        node_names.add(pi_name_prefixed)
        node_types[pi_name_prefixed] = 'PI'

    # 2nd pass: Build edges
    edges = []
    for _, gate_name, ports_str in gate_insts:
        target_node = f"{design_name}_{gate_name.strip()}"
        ports = [p.strip() for p in ports_str.split(',')]
        for wire_raw in ports[1:]:
            wire_match = re.search(r'\(?([\w\[\]\']+)\)?', wire_raw)
            if not wire_match: continue
            wire = wire_match.group(1)
            wire_base_name = re.match(r'(\w+)', wire).group(1)
            source_node = wire_to_driver.get(wire) or wire_to_driver.get(wire_base_name) or \
                          (f"{design_name}_{wire_base_name}" if wire_base_name in primary_inputs else None)
            if source_node: edges.append((source_node, target_node))
    
    trojan_nodes_prefixed = {f"{design_name}_{g}" for g in trojan_local_gates}
    return node_names, edges, node_types, trojan_nodes_prefixed


def create_split_disjoint_graph(circuit_files, output_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Processes and merges circuits, then splits them by circuit into train/val/test roles.
    """
    
    circuit_to_nodes = defaultdict(set)
    all_edges = []
    all_node_types = {}
    all_trojan_nodes = set()

    print("--- Step 1: Parsing all circuits into memory ---")
    for verilog_path, trojan_path in circuit_files:
        design_name = os.path.splitext(os.path.basename(verilog_path))[0]
        print(f"  - Parsing: {design_name}")
        result = parse_single_circuit(verilog_path, trojan_path, design_name)
        if result:
            nodes, edges, types, trojans = result
            circuit_to_nodes[design_name].update(nodes)
            all_edges.extend(edges)
            all_node_types.update(types)
            all_trojan_nodes.update(trojans)

    all_nodes = {node for nodes in circuit_to_nodes.values() for node in nodes}
    sorted_nodes = sorted(list(all_nodes))
    node_to_idx = {name: i for i, name in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)
    print(f"-> Parsed {len(circuit_to_nodes)} circuits with a total of {num_nodes} unique nodes.")

    print("\n--- Step 2: Splitting circuits into Train/Validation/Test sets ---")
    design_names = list(circuit_to_nodes.keys())
    random.shuffle(design_names)

    tr_end = int(split_ratio[0] * len(design_names))
    va_end = tr_end + int(split_ratio[1] * len(design_names))
    
    train_designs = design_names[:tr_end]
    val_designs = design_names[tr_end:va_end]
    test_designs = design_names[va_end:]

    role = {'tr': [], 'va': [], 'te': []}
    for d in train_designs:
        role['tr'].extend([node_to_idx[n] for n in circuit_to_nodes[d]])
    for d in val_designs:
        role['va'].extend([node_to_idx[n] for n in circuit_to_nodes[d]])
    for d in test_designs:
        role['te'].extend([node_to_idx[n] for n in circuit_to_nodes[d]])
    
    print(f"-> Split: {len(train_designs)} train, {len(val_designs)} val, {len(test_designs)} test circuits.")
    print(f"-> Nodes: {len(role['tr'])} train, {len(role['va'])} val, {len(role['te'])} test nodes.")

    print("\n--- Step 3: Assembling final graph files ---")
    
    # Create class_map and feats for ALL nodes
    class_map = {node_to_idx[name]: (1 if name in all_trojan_nodes else 0) for name in sorted_nodes}
    unique_types = sorted(list(set(all_node_types.values())))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    feats = np.zeros((num_nodes, len(unique_types)), dtype=np.float32)
    for i, name in enumerate(sorted_nodes):
        feats[i, type_to_idx[all_node_types[name]]] = 1.0

    # Create adj_full with ALL edges
    all_edge_indices = [(node_to_idx[u], node_to_idx[v]) for u, v in all_edges]
    rows, cols = zip(*all_edge_indices)
    data = np.ones(len(rows), dtype=np.uint8)
    adj_full = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    print(f"-> adj_full created with {adj_full.nnz} edges.")

    # Create adj_train with ONLY training edges
    train_idx_set = set(role['tr'])
    train_edge_indices = [(u, v) for u, v in all_edge_indices if u in train_idx_set and v in train_idx_set]
    if train_edge_indices:
        rows_tr, cols_tr = zip(*train_edge_indices)
        data_tr = np.ones(len(rows_tr), dtype=np.uint8)
        adj_train = sp.csr_matrix((data_tr, (rows_tr, cols_tr)), shape=(num_nodes, num_nodes))
    else:
        adj_train = sp.csr_matrix((num_nodes, num_nodes), dtype=np.uint8)
    print(f"-> adj_train created with {adj_train.nnz} edges.")

    print("\n--- Step 4: Saving all files ---")
    os.makedirs(output_dir, exist_ok=True)
    sp.save_npz(os.path.join(output_dir, 'adj_full.npz'), adj_full)
    sp.save_npz(os.path.join(output_dir, 'adj_train.npz'), adj_train)
    np.save(os.path.join(output_dir, 'feats.npy'), feats)
    with open(os.path.join(output_dir, 'role.json'), 'w') as f:
        json.dump(role, f)
    with open(os.path.join(output_dir, 'class_map.json'), 'w') as f:
        json.dump(class_map, f)
    print("Done. ✨")

if __name__ == '__main__':
    design_num = 20
    gatelist_base = "cadcontest/release2/trojan_design"
    label_base = "cadcontest/reference"
    files_to_process = []
    for i in range(design_num):
        files_to_process.append((os.path.join(gatelist_base, f"design{i}.v"), os.path.join(label_base, f"result{i}.txt")))

    # --- Execute the merging script ---
    # e.g., 70% train, 15% val, 15% test of the circuits
    create_split_disjoint_graph(
        circuit_files=files_to_process,
        output_dir='split_merged_graph_output',
        split_ratio=(0.7, 0.15, 0.15)
    )
