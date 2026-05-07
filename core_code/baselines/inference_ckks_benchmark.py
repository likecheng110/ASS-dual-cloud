import torch
import time
import tenseal as ts
import numpy as np
import sys
import os
import json


def _error_payload(message):
    return {
        'Acc': None,
        'Time': None,
        'Comm': f'Error: {message}',
        'Layer': None,
        'Status': 'FAILED',
        'Error': str(message),
    }


def run_ckks_micro(task_name, model_path):
    # Log file relative to execution dir or absolute? 
    # Let's use current dir for log
    # with open('ckks_micro.log', 'w') as f: f.write("Starting...\n")
    try:
        # 1. Setup CKKS Context
        # Try even simpler context for stability
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=4096,
            coeff_mod_bit_sizes=[30, 20, 30] # Lower bits
        )
        context.global_scale = 2**20
        context.generate_galois_keys()
        
        # with open('ckks_micro.log', 'a') as f: f.write("Context OK.\n")
        
        # 2. Load Model Structure
        state_dict = torch.load(model_path, map_location='cpu')
        w1 = state_dict['fc1.weight'].numpy().T # (Input, Hidden)
        input_dim, hidden_dim = w1.shape
        
        if 'fc2.weight' in state_dict:
            w2 = state_dict['fc2.weight'].numpy().T # (Hidden, Output)
            output_dim = w2.shape[1]
        else:
            output_dim = 0
            
        # with open('ckks_micro.log', 'a') as f: f.write(f"Dims: {input_dim}, {hidden_dim}, {output_dim}\n")
            
        # 3. Micro-benchmark Atoms
        
        # A. Encryption
        dummy_input = np.random.rand(input_dim)
        t0 = time.perf_counter_ns()
        try:
            enc_x = ts.ckks_vector(context, dummy_input)
        except Exception as e:
            # with open('ckks_micro.log', 'a') as f: f.write(f"Enc Error: {e}\n")
            return None
        t1 = time.perf_counter_ns()
        t_enc = (t1 - t0) / 1e9
        
        # with open('ckks_micro.log', 'a') as f: f.write(f"Enc Time: {t_enc}\n")
        
        # B. Linear Atom (Dot Product)
        # 1 input neuron * 1 input neuron (scalar mult) or 1 vector * 1 vector?
        # TenSEAL dot: vector * vector -> scalar (in ciphertext)
        # We need to simulate: enc_vector(N) dot plain_col(N) -> enc_scalar
        
        dummy_col = np.random.rand(input_dim)
        t0 = time.perf_counter_ns()
        try:
            # Run 1 dot product
            res = enc_x.dot(dummy_col)
        except Exception as e:
            # with open('ckks_micro.log', 'a') as f: f.write(f"Dot Error: {e}\n")
            return None
        t1 = time.perf_counter_ns()
        t_linear_atom = (t1 - t0) / 1e9
        
        # with open('ckks_micro.log', 'a') as f: f.write(f"Linear Atom: {t_linear_atom}\n")
        
        # C. Activation Atom (Square)
        # Square the RESULT of the dot product (which is a ciphertext of size 1? or size N?)
        # TenSEAL dot returns a ciphertext.
        # Let's square it.
        t0 = time.perf_counter_ns()
        try:
            # Maybe use a fresh vector?
            # res is likely a ciphertext with 1 slot filled?
            # Or N slots?
            # Square on windows might be problematic with memory alignment.
            # Try square a fresh small vector.
            dummy_small = ts.ckks_vector(context, [0.1])
            dummy_small.square_()
            t_act_atom_real = (time.perf_counter_ns() - t0) / 1e9
        except Exception as e:
            # with open('ckks_micro.log', 'a') as f: f.write(f"Square Error: {e}\n")
            return None
        
        # Use the fresh measurement
        t_act_atom = t_act_atom_real
        
        # with open('ckks_micro.log', 'a') as f: f.write(f"Act Atom: {t_act_atom}\n")
        
        # D. Layer 2 Atom (Scalar Mult)
        # Maybe dummy_small.mul crashes?
        # Try a fresh vector again?
        try:
            d2 = ts.ckks_vector(context, [0.5])
            t0 = time.perf_counter_ns()
            d2.mul(0.5)
            t_scalar_mul = (time.perf_counter_ns() - t0) / 1e9
        except Exception as e:
            # with open('ckks_micro.log', 'a') as f: f.write(f"Mul Error: {e}\n")
            return None
            
        # with open('ckks_micro.log', 'a') as f: f.write(f"Scalar Mul Atom: {t_scalar_mul}\n")
        
        # E. Decryption
        t0 = time.perf_counter_ns()
        try:
            # res.decrypt()
            dummy_small.decrypt()
            t_dec = (time.perf_counter_ns() - t0) / 1e9
        except Exception as e:
            # with open('ckks_micro.log', 'a') as f: f.write(f"Dec Error: {e}\n")
            return None
        
        # with open('ckks_micro.log', 'a') as f: f.write(f"Dec Time: {t_dec}\n")
        
        # 4. Projection
        t_layer1 = hidden_dim * t_linear_atom
        t_relu = hidden_dim * t_act_atom
        t_layer2 = output_dim * hidden_dim * t_scalar_mul if output_dim > 0 else 0
        
        total_time = t_enc + t_layer1 + t_relu + t_layer2 + t_dec
        
        # Accuracy: Return None to indicate not evaluated due to micro-benchmark nature
        acc = None
        
        return {
            'Acc': acc,
            'Time': total_time,
            'Comm': 'High',
            'Layer': {
                'Linear': t_layer1 + t_layer2,
                'ReLU': t_relu,
                'EncDec': t_enc + t_dec
            }
        }
        
    except Exception as e:
        # with open('ckks_micro.log', 'a') as f: f.write(f"General Error: {e}\n")
        return _error_payload(str(e))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        task = "MNIST"
        model = "models/mnist_mlp.pth"
    else:
        task = sys.argv[1]
        model = sys.argv[2]
    
    res = run_ckks_micro(task, model)
    if res is None:
        res = _error_payload("CKKS micro-benchmark returned no result")
    print("JSON_START")
    print(json.dumps(res))
