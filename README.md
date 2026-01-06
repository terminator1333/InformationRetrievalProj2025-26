```text
 __      __  ____    __       ____     _____            ____        ______  _____       _____   __  __  ____                       
/\ \  __/\ \/\  _`\ /\ \     /\  _`\  /\  __`\  /'\_/`\/\  _`\     /\__  _\/\  __`\    /\  __`\/\ \/\ \/\  _`\                     
\ \ \/\ \ \ \ \ \L\_\ \ \    \ \ \/\_\\ \ \/\ \/\      \ \ \L\_\   \/_/\ \/\ \ \/\ \   \ \ \/\ \ \ \ \ \ \ \L\ \                   
 \ \ \ \ \ \ \ \  _\L\ \ \  __\ \ \/_/_\ \ \ \ \ \ \__\ \ \  _\L      \ \ \ \ \ \ \ \   \ \ \ \ \ \ \ \ \ \ ,  /                   
  \ \ \_/ \_\ \ \ \L\ \ \ \L\ \\ \ \L\ \\ \ \_\ \ \ \_/\ \ \ \L\ \     \ \ \ \ \ \_\ \   \ \ \_\ \ \ \_\ \ \ \\ \                  
   \ `\___x___/\ \____/\ \____/ \ \____/ \ \_____\ \_\\ \_\ \____/      \ \_\ \ \_____\   \ \_____\ \_____\ \_\ \_\                
    '\/__//__/  \/___/  \/___/   \/___/   \/_____/\/_/ \/_/\/___/        \/_/  \/_____/    \/_____/\/_____/\/_/\/ /                
                                                                                                                                   
                                                                                                                                   
            ______   __  __  ____    _____       ____    ____    ______  ____    ______   ____    __  __  ______  __               
           /\__  _\ /\ \/\ \/\  _`\ /\  __`\    /\  _`\ /\  _`\ /\__  _\/\  _`\ /\__  _\ /\  _`\ /\ \/\ \/\  _  \/\ \              
           \/_/\ \/ \ \ `\\ \ \ \L\_\ \ \/\ \   \ \ \L\ \ \ \L\_\/_/\ \/\ \ \L\ \/_/\ \/ \ \ \L\_\ \ \ \ \ \ \L\ \ \ \             
              \ \ \  \ \ , ` \ \  _\/\ \ \ \ \   \ \ ,  /\ \  _\L  \ \ \ \ \ ,  /  \ \ \  \ \  _\L\ \ \ \ \ \  __ \ \ \  __        
               \_\ \__\ \ \`\ \ \ \/  \ \ \_\ \   \ \ \\ \\ \ \L\ \ \ \ \ \ \ \\ \  \_\ \__\ \ \L\ \ \ \_/ \ \ \/\ \ \ \L\ \       
               /\_____\\ \_\ \_\ \_\   \ \_____\   \ \_\ \_\ \____/  \ \_\ \ \_\ \_\/\_____\\ \____/\ `\___/\ \_\ \_\ \____/       
               \/_____/ \/_/\/_/\/_/    \/_____/    \/_/\/ /\/___/    \/_/  \/_/\/ /\/_____/ \/___/  `\/__/  \/_/\/_/\/___/        
                                                                                                                                   
                                                                                                                                   
                           ____    ____    _____    _____  ____    ______                                                          
                          /\  _`\ /\  _`\ /\  __`\ /\___ \/\  _`\ /\__  _\                                                         
                          \ \ \L\ \ \ \L\ \ \ \/\ \\/__/\ \ \ \L\_\/_/\ \/                                                         
                           \ \ ,__/\ \ ,  /\ \ \ \ \  _\ \ \ \  _\L  \ \ \                                                         
                            \ \ \/  \ \ \\ \\ \ \_\ \/\ \_\ \ \ \L\ \ \ \ \                                                        
                             \ \_\   \ \_\ \_\ \_____\ \____/\ \____/  \ \_\                                                       
                              \/_/    \/_/\/ /\/_____/\/___/  \/___/    \/_/ 

                                                                                                                                                                 
                            _..._
                          \_.._ `'-.,--,
                           '-._'-.  `\a\\
                               '. `_.' (|
                                 `7    ||
                                 /   .' |
                                /_.-'  ,J
                               /         \
                              ||   /      ;
                   _..        ||  |       |  /`\.-.
                 .' _ `\      `\  \       |  \_/__/
                /  /e)-,\       '. \      /.-` .'\
               /  |  ,_ |        /\ `;_.-'_.-'`\_/
              /   '-(-.)/        \_;(((_.-;
            .'--.   \  `       .(((_,;`'.  \
           /    `\   |   _.--'`__.'  `\  '-;\
         /`       |  /.-'  .--'        '._.'\\
       .'        ;  /__.-'`             |  \ |
     .'`-'_     /_.')))                  \_\,_/
    / -'_.'---;`'-)))
   (__.'/   /` .'`
    (_.'/ /` /`
      _|.' /`
   .-` __.'|
    .-'||  |
       \_`/
         `                                                                                                                                                                   
```
This is our project for the Information Retrieval course, where we built a running search engine both in colab and in GCP via a VM.

The search engine is viable for the entire English Wikipedia corpus up to Aug 2021




A bit on the given files in this repo: 


- GCP/
  - instal_gcs.sh -> when creating the Inverted Indexes for the Title, Body, and Anchor we needed the worker nodes to be updated to the required google-cloud-storage version. We added this script to GCP when initialising the cluster
  - inverted_index_gcp.py -> helper script, has the Inverted Index class used when creating the Inverted Index and in the VM
  - gcp_create_inverted_indexes.ipynb -> script for creating the Inverted Indexes for the Title, Body, and Anchor, in addition to creating the Pagerank and doc-title mappings. Was run on a cluster in GCP
  - create_pageviews.ipynb -> script for creating pageviews data in the bucket. Ran on Google Colab
  - run_frontend_in_gcp.sh -> script for initiating VM
  - startup_script_gcp.sh -> script for helping run_frontend_in_gcp initiate VM
  - change_vm_to_venv.sh -> Since all the libraries are stored in virtual environment, this is a script to initialise the venv activation automatically
  - optimize_weights.py -> Doing a grid search to find best hyperparams. Using 20/30 of given queries
  - run_testing_gcp.py -> Script for testing all given queries
- Colab/
  - run_frontend_in_colab.ipynb -> script for making sure the server works and testing different approaches
- queries_train.json -> given queries to train. We mostly used a 20/10 split (20 for calibrating hyperparameters and 10 for testing)
- search_backend.py -> main functionality of the project. Implements the different search functions
- search_frontend.py -> frontend allowing communication between flask app and backend


How are the indexes created: 
- In the search frontend script, we create an instance of the backend class. This downloads all the files to the local machine, with the efforts of saving search time. This is done using the client feature in google-cloud-console


How to use (from scratch): 
- Create the inverted indexes, pageranks and pageviews
  - For Colab:
    - Upload search_backend.py, search_frontend.py, queries_train.py, run_frontend_in_colab.ipynb
    - Run run_frontent_in_colab.ipynb
  - For GCP:
    - Initialise VM using run_frontend_in_gcp, startup_script_gcp.sh, change_vm_to_venv.sh
    - Upload search_backend.py, search_frontend.py, queries_train.py, run_testing_gcp.py
    - Run search_frontend.py (downloads the files locally)
    - Run run_testing_gcp.py
    - FOR SPECIFIC QUERIES PLEASE NAVIGATE TO http://localhost:8080/search?query=your+query (remove 'your+query' with the actual queries, but replace spaces with pluses('+')


How to use the VM:
- Upload search_backend.py, search_frontend.py, queries_train.py, run_testing_gcp.py
- Run search_frontend.py
- Run run_testing_gcp.py


<img width="2816" height="1536" alt="Gemini_Generated_Image_9fzh2i9fzh2i9fzh" src="https://github.com/user-attachments/assets/06d415ee-504b-41b0-b83b-935a80854a03" />
