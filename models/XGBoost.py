# XGBoost.py (simplified)
import logging
import time
import sys
from pathlib import Path

# Setup path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from processing.model_utils import (
    process_target, save_results, save_tuning_params, 
    save_pruning_results, save_model_and_data
)

def main():
    """Main function to run the XGBoost pipeline."""
    model_name = 'XGBoost'
    
    # Config
    targets = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18']
    poly_degree = 0
    test_size = 0.2
    
    # Mode selection
    tuning = False
    n_trials = 50
    prune = False
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Track timing
    start_time = time.time()
    results = []
    params_dict = {}
    pruning_results = []
    
    logger.info(f"Starting {model_name} pipeline")
    
    # Process each target
    for target in targets:
        logger.info("")
        logger.info(f"Processing target: {target}")
        best_params, plotting_data, result, target_pruning_results = process_target(
            model_name, target, poly_degree, test_size, tuning, n_trials, prune
        )
        
        # Store results
        results.append(result)
        if tuning:
            params_dict[target] = best_params
        if prune and target_pruning_results:
            pruning_results.append(target_pruning_results)
        
        # Save model and data for inference mode
        if not tuning and not prune:
            save_model_and_data(model_name, target, best_params, plotting_data)
    
    # Calculate and record total time
    elapsed_time = time.time() - start_time
    logger.info(f"Total elapsed time: {elapsed_time:.4f} seconds")
    results.append({'Elapsed time': elapsed_time})
    
    # Save results based on mode
    if tuning:
        save_tuning_params(model_name, params_dict)
        save_results(model_name, results, tuning=True, n_trials=n_trials)
    elif prune:
        save_pruning_results(model_name, pruning_results, targets)
    else:
        save_results(model_name, results)
    
    logger.info("")
    logger.info(f"{model_name} pipeline completed successfully")

if __name__ == '__main__':
    main()