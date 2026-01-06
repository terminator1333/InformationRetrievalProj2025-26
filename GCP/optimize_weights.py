import json
import statistics
import random
from search_backend import SearchBackend

# ================= CONFIGURATION =================
QUERY_FILE = 'queries_train.json'
NUM_QUERIES = 20  # Limit to 20 queries for speed

# Define the Grid (The values to test)
# 4 x 3 x 3 x 3 = 108 combinations
GRID = {
    'w_title':  [1, 5, 10, 20],   # Titles are usually very important
    'w_anchor': [0.5, 2, 5],      # Anchor text is good but noisy
    'w_body':   [1.0],            # Keep body fixed as the baseline (1.0)
    'w_pr':     [0.1, 1, 5],      # PageRank weights
    'w_pv':     [0.1, 1, 5]       # PageView weights
}
# =================================================

def run_grid_search():
    # 1. Load Data
    try:
        with open(QUERY_FILE, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {QUERY_FILE} not found.")
        return

    # 2. Select 20 Queries
    # We convert to a list of items and take the first 20
    # (Or use random.sample if you want random ones)
    queries_subset = list(all_data.items())[:NUM_QUERIES]
    
    print(f"Loaded {len(queries_subset)} queries for optimization.")
    print("Initializing Backend...")
    backend = SearchBackend()
    print("Backend ready. Starting Grid Search...\n")

    print(f"{'Title':<6} | {'Anch':<6} | {'PR':<6} | {'PV':<6} | {'MAP@10':<8} | {'Recall':<8}")
    print("-" * 65)

    best_score = -1
    best_params = {}

    # 3. Iterate the Grid
    for t in GRID['w_title']:
        for a in GRID['w_anchor']:
            for b in GRID['w_body']:
                for pr in GRID['w_pr']:
                    for pv in GRID['w_pv']:
                        
                        precisions = []
                        recalls = []

                        # Run the 20 queries
                        for query, true_ids in queries_subset:
                            true_ids = set(map(str, true_ids)) # Faster lookups
                            
                            # CALL SEARCH WITH CURRENT WEIGHTS
                            results = backend.search(
                                query, 
                                w_title=t, 
                                w_anchor=a, 
                                w_body=b, 
                                w_pr=pr, 
                                w_pv=pv
                            )
                            
                            # Extract IDs
                            retrieved_ids = [str(x[0]) for x in results]
                            
                            # Calculate Precision@10
                            top_10 = retrieved_ids[:10]
                            relevant_10 = len(set(top_10).intersection(true_ids))
                            precisions.append(relevant_10 / 10 if top_10 else 0)
                            
                            # Calculate Recall
                            relevant_total = len(set(retrieved_ids).intersection(true_ids))
                            recalls.append(relevant_total / len(true_ids) if true_ids else 0)

                        # Average Metrics
                        avg_p = statistics.mean(precisions)
                        avg_r = statistics.mean(recalls)

                        # Print row
                        print(f"{t:<6} | {a:<6} | {pr:<6} | {pv:<6} | {avg_p:.4f}   | {avg_r:.4f}")

                        # Check if winner
                        # We optimize for Precision, but you can change logic to (avg_p + avg_r)
                        if avg_p > best_score:
                            best_score = avg_p
                            best_params = {
                                'w_title': t, 
                                'w_anchor': a, 
                                'w_body': b, 
                                'w_pr': pr, 
                                'w_pv': pv
                            }

    print("\n" + "="*40)
    print(f"WINNER: MAP@10 = {best_score:.4f}")
    print("Params:", best_params)
    print("="*40)

if __name__ == "__main__":
    run_grid_search()