import json
import requests
import time
import statistics

# ================= CONFIGURATION =================
URL = 'http://127.0.0.1:8080/search' 
QUERY_FILE = 'queries_train.json'
# =================================================

def calculate_metrics(retrieved_ids, true_ids):
    """ Calculates Precision@K and Recall """
    if not retrieved_ids:
        return 0, 0
    
    # K is the number of results returned (usually 10, 20, or 100)
    K = len(retrieved_ids) 
    
    # Intersection: How many retrieved docs are actually relevant?
    relevant_retrieved = len(set(retrieved_ids).intersection(set(true_ids)))
    
    precision = relevant_retrieved / K
    recall = relevant_retrieved / len(true_ids) if len(true_ids) > 0 else 0
    
    return precision, recall

def run_benchmark():
    try:
        with open(QUERY_FILE, 'r') as f:
            queries_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {QUERY_FILE}")
        return

    print(f"Loaded {len(queries_data)} queries. Starting benchmark on {URL}...\n")
    print(f"{'Query':<30} | {'Lat (ms)':<10} | {'Prec':<6} | {'Recall':<6}")
    print("-" * 65)

    latencies = []
    precisions = []
    recalls = []

    for query, true_ids in queries_data.items():
        # Normalize true_ids to strings to ensure matching works
        true_ids = [str(x) for x in true_ids]
        
        start_time = time.time()
        try:
            # Send request to your Flask server
            response = requests.get(URL, params={'query': query})
            
            # Stop timer
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            latencies.append(duration_ms)

            if response.status_code == 200:
                results = response.json()
                
                # Handling different return formats:
                # 1. If it returns a list of IDs: [1, 2, 3]
                # 2. If it returns a list of tuples: [[1, "title"], [2, "title"]]
                retrieved_ids = []
                if isinstance(results, list):
                    for item in results:
                        if isinstance(item, list) or isinstance(item, tuple):
                            retrieved_ids.append(str(item[0])) # Take first element (ID)
                        else:
                            retrieved_ids.append(str(item)) # Take raw ID
                
                p, r = calculate_metrics(retrieved_ids, true_ids)
                precisions.append(p)
                recalls.append(r)
                
                # Print short summary for this query (truncated)
                print(f"{query[:28]:<30} | {duration_ms:>8.2f} | {p:>6.2f} | {r:>6.2f}")
            
            else:
                print(f"{query[:28]:<30} | ERROR {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("\nCRITICAL ERROR: Could not connect to server.")
            print("Is it running? (python3 search_frontend.py)")
            return

    # FINAL SUMMARY
    print("\n" + "="*30)
    print("       BENCHMARK RESULTS       ")
    print("="*30)
    if latencies:
        print(f"Total Queries:    {len(latencies)}")
        print(f"Avg Latency:      {statistics.mean(latencies):.2f} ms")
        print(f"Avg Precision:    {statistics.mean(precisions):.4f}")
        print(f"Avg Recall:       {statistics.mean(recalls):.4f}")
    else:
        print("No queries ran successfully.")

if __name__ == "__main__":
    run_benchmark()
