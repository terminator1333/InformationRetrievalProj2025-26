import json
import requests
import time
import statistics

# ================= CONFIGURATION =================
URL = 'http://127.0.0.1:8080/search' 
QUERY_FILE = 'queries_train.json'
USE_COS_SIM = True  # Toggle True/False
# =================================================

def calculate_precision_at_k(retrieved_ids, true_ids, k):
    """
    Calculates Precision at rank K.
    P@K = (Relevant items in top K) / K
    """
    if not retrieved_ids:
        return 0.0
    
    # Slice the list to the top K results
    top_k = retrieved_ids[:k]
    
    # Count how many are in the true set (intersection)
    # We convert true_ids to a set for O(1) lookups
    true_set = set(true_ids)
    relevant_hits = [doc_id for doc_id in top_k if doc_id in true_set]
    
    return len(relevant_hits) / k

def calculate_f1_at_k(retrieved_ids, true_ids, k):
    """
    Calculates F1 score at rank K.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    if not true_ids:
        return 0.0
        
    # Calculate P@K
    p_at_k = calculate_precision_at_k(retrieved_ids, true_ids, k)
    
    # Calculate Recall@K
    # Recall@K = (Relevant items in top K) / (Total Relevant items)
    top_k = retrieved_ids[:k]
    true_set = set(true_ids)
    relevant_hits = len([doc_id for doc_id in top_k if doc_id in true_set])
    r_at_k = relevant_hits / len(true_ids)
    
    if (p_at_k + r_at_k) == 0:
        return 0.0
        
    return 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)

def run_benchmark():
    try:
        with open(QUERY_FILE, 'r') as f:
            queries_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {QUERY_FILE}")
        return

    mode_str = " (Cosine Similarity)" if USE_COS_SIM else " (BM25/Default)"
    print(f"Loaded {len(queries_data)} queries. Starting benchmark on {URL}{mode_str}...\n")
    
    # Table Header
    print(f"{'Query':<25} | {'Lat(ms)':<8} | {'P@5':<6} | {'P@10':<6} | {'F1@30':<6}")
    print("-" * 70)

    # Lists to store metrics for averaging
    latencies = []
    p5_scores = []
    p10_scores = []
    f1_30_scores = []

    for query, true_ids in queries_data.items():
        # Ensure true_ids are strings for comparison
        true_ids = [str(x) for x in true_ids]
        
        start_time = time.time()
        try:
            params = {
                'query': query,
                'use_cos_sim': 'true' if USE_COS_SIM else 'false' 
            }
            response = requests.get(URL, params=params)
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            latencies.append(duration_ms)

            if response.status_code == 200:
                results = response.json()
                
                # normalize results to a flat list of IDs
                retrieved_ids = []
                if isinstance(results, list):
                    for item in results:
                        if isinstance(item, list) or isinstance(item, tuple):
                            retrieved_ids.append(str(item[0])) 
                        else:
                            retrieved_ids.append(str(item)) 
                
                # --- CALC METRICS ---
                p5 = calculate_precision_at_k(retrieved_ids, true_ids, k=5)
                p10 = calculate_precision_at_k(retrieved_ids, true_ids, k=10)
                f1_30 = calculate_f1_at_k(retrieved_ids, true_ids, k=30)
                
                # Store for averages
                p5_scores.append(p5)
                p10_scores.append(p10)
                f1_30_scores.append(f1_30)
                
                # Print row
                print(f"{query[:23]:<25} | {duration_ms:>8.2f} | {p5:>6.2f} | {p10:>6.2f} | {f1_30:>6.2f}")
            
            else:
                print(f"{query[:23]:<25} | ERROR {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("\nCRITICAL ERROR: Could not connect to server.")
            return

    # --- FINAL AVERAGES ---
    print("\n" + "="*35)
    print("       BENCHMARK RESULTS        ")
    print("="*35)
    
    if latencies:
        print(f"Total Queries:     {len(latencies)}")
        print(f"Avg Latency:       {statistics.mean(latencies):.2f} ms")
        print("-" * 35)
        print(f"MAP@5 (Avg P@5):   {statistics.mean(p5_scores):.4f}")
        print(f"MAP@10 (Avg P@10): {statistics.mean(p10_scores):.4f}")
        print(f"Avg F1@30:         {statistics.mean(f1_30_scores):.4f}")
    else:
        print("No queries ran successfully.")

if __name__ == "__main__":
    run_benchmark()