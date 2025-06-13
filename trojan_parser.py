import re
import numpy as np
import os
from scipy.sparse import csr_matrix, lil_matrix, save_npz
import json
import glob

def read_trojan_gates(result_file):
    """Read trojan gate labels from result file"""
    trojan_gates = set()
    with open(result_file, 'r') as f:
        lines = f.readlines()
        reading_gates = False
        for line in lines:
            line = line.strip()
            if line == 'TROJAN_GATES':
                reading_gates = True
                continue
            elif line == 'END_TROJAN_GATES':
                reading_gates = False
                continue
            elif reading_gates:
                trojan_gates.add(line)
    return trojan_gates

def parse_verilog(verilog_file):
    """Parse verilog file to extract gates and connections"""
    gates = {}  # gate_name -> (gate_type, inputs, output)
    with open(verilog_file, 'r') as f:
        content = f.read()
        
        # Extract all gate instances
        gate_pattern = r'(\w+)\s+(\w+)\s*\(([^;]+)\);'
        for match in re.finditer(gate_pattern, content):
            gate_type = match.group(1)
            gate_name = match.group(2)
            connections = match.group(3)
            
            # Parse connections
            conn_parts = [p.strip() for p in connections.split(',')]
            output = conn_parts[0].strip()
            inputs = [p.strip() for p in conn_parts[1:]]
            
            gates[gate_name] = (gate_type, inputs, output)
            
    return gates

def build_graph(gates, trojan_gates):
    """Build graph from gates and generate features"""
    # Create node mapping
    nodes = {}  # node_name -> node_id
    node_id = 0
    
    # First pass: assign IDs to all nodes
    for gate_name, (gate_type, inputs, output) in gates.items():
        if gate_name not in nodes:
            nodes[gate_name] = node_id
            node_id += 1
        if output not in nodes:
            nodes[output] = node_id
            node_id += 1
        for inp in inputs:
            if inp not in nodes:
                nodes[inp] = node_id
                node_id += 1
    
    # Create adjacency matrix
    n_nodes = len(nodes)
    adj = lil_matrix((n_nodes, n_nodes), dtype=bool)
    
    # Add edges
    for gate_name, (gate_type, inputs, output) in gates.items():
        gate_id = nodes[gate_name]
        output_id = nodes[output]
        adj[gate_id, output_id] = True
        adj[output_id, gate_id] = True
        for inp in inputs:
            inp_id = nodes[inp]
            adj[gate_id, inp_id] = True
            adj[inp_id, gate_id] = True
    
    # Create features
    gate_types = ['nor', 'not', 'and', 'or', 'xor', 'xnor', 'dff']
    n_gate_types = len(gate_types)
    n_features = n_gate_types + 4  # gate type one-hot + 4 additional features
    
    features = np.zeros((n_nodes, n_features))
    
    # Set gate type features
    for gate_name, (gate_type, inputs, output) in gates.items():
        gate_id = nodes[gate_name]
        if gate_type in gate_types:
            type_idx = gate_types.index(gate_type)
            features[gate_id, type_idx] = 1
    
    # Set additional features
    for node_name, node_id in nodes.items():
        # Feature 1: Number of inputs (normalized)
        if node_name in gates:
            n_inputs = len(gates[node_name][1])
            features[node_id, n_gate_types] = n_inputs / 10.0
        
        # Feature 2: Number of neighbors (normalized)
        n_neighbors = adj[node_id].sum()
        features[node_id, n_gate_types + 1] = n_neighbors / 20.0
        
        # Feature 3: Is input node
        is_input = any(node_name == inp for _, (_, inputs, _) in gates.items() for inp in inputs)
        features[node_id, n_gate_types + 2] = 1.0 if is_input else 0.0
        
        # Feature 4: Is output node
        is_output = any(node_name == output for _, (_, _, output) in gates.items())
        features[node_id, n_gate_types + 3] = 1.0 if is_output else 0.0
    
    # Create class map
    class_map = {}
    for gate_name, node_id in nodes.items():
        if gate_name in trojan_gates:
            class_map[str(node_id)] = 1
        else:
            class_map[str(node_id)] = 0
    
    # Create role map (all nodes for training)
    role = {
        'tr': list(range(n_nodes)),
        'va': [],
        'te': []
    }
    
    return adj, features, class_map, role

def process_design(design_file, result_file, output_dir):
    """Process a single design file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read trojan gates
    trojan_gates = read_trojan_gates(result_file)
    
    # Parse verilog
    gates = parse_verilog(design_file)
    
    # Build graph
    adj, features, class_map, role = build_graph(gates, trojan_gates)
    
    # Save outputs
    adj_full = adj.tocsr()
    adj_train = adj.tocsr()  # Same as adj_full since we're using all nodes for training
    
    np.save(os.path.join(output_dir, 'feats.npy'), features)
    with open(os.path.join(output_dir, 'class_map.json'), 'w') as f:
        json.dump(class_map, f)
    with open(os.path.join(output_dir, 'role.json'), 'w') as f:
        json.dump(role, f)
    
    save_npz(os.path.join(output_dir, 'adj_full.npz'), adj_full)
    save_npz(os.path.join(output_dir, 'adj_train.npz'), adj_train)

def main():
    # Process all designs
    design_dir = 'cadcontest/release2/trojan_design'
    result_dir = 'cadcontest/reference'
    
    for i in range(20):
        design_file = os.path.join(design_dir, f'design{i}.v')
        result_file = os.path.join(result_dir, f'result{i}.txt')
        output_dir = f'processed/design{i}'
        
        print(f'Processing design {i}...')
        process_design(design_file, result_file, output_dir)
        print(f'Done processing design {i}')

if __name__ == '__main__':
    main() 