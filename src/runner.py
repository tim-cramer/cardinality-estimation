import time
import pandas as pd
import random
from src.algorithms import TrueCardinality, HyperLogLog, Recordinality, PCSA

def compare_algos(stream, configs, stream_name="Data", trials=5):
    """
    configs: List of tuples/objects describing what to run.
    Example: [('HLL', 10), ('REC', 64), ('PCSA', 10)]
    """
    
    # Ground Truth
    t_algo = TrueCardinality()
    for x in stream: t_algo.add(x)
    true_n = t_algo.estimate()
    
    data = []
    
    for _ in range(trials):
        # Generate a random seed for this specific trial to ensure statistical independence
        current_seed = random.randint(0, 1000000)

        for algo_type, param in configs:
            if algo_type == 'HLL':
                algo = HyperLogLog(b=param, seed=current_seed)
                x_val = 2**param 

            elif algo_type == 'REC':
                algo = Recordinality(k=param, seed=current_seed)
                x_val = param

            elif algo_type == 'REC_NoHash':
                algo = Recordinality(k=param, use_hash=False, seed=current_seed)
                x_val = param

            elif algo_type == 'PCSA':
                algo = PCSA(b=param, seed=current_seed)
                x_val = 2**param
                
            # Run
            start = time.time()
            for item in stream:
                algo.add(item)
            dur = time.time() - start
            
            est = algo.estimate()
            err = abs(est - true_n) / true_n
            
            data.append({
                "Algorithm": algo.name,
                "Type": algo_type,
                "Param": param,
                "Memory_Scale": x_val,
                "Estimate": est,
                "True_Count": true_n,
                "Rel_Error": err,
                "Duration": dur,
                "Stream": stream_name
            })
            
    return pd.DataFrame(data)