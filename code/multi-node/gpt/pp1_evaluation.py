import json
import os
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

def get_configuration():
    """Parse command line arguments and build configuration"""
    parser = argparse.ArgumentParser(description='Parallelization coordinator evaluation (Views)')
    parser.add_argument('-d', '--directory', required=True,
                       help='Base directory path (e.g., ./data/in-domain/bin0/)')
    parser.add_argument('-f', '--file', required=True,
                       help='Path to the feature extracted data file for evaluation')
    
    args = parser.parse_args()
    
    # Build configuration from the two main options
    base_dir = args.directory.rstrip('/')  # Remove trailing slash if present
    
    config = {
        'INPUT_FILE': args.file,
        'OUTPUT_DIR': f'{base_dir}/evaluation_result'
    }
    
    return config

def load_parallel_agent_data(file_path: str) -> list:
    """Load the parallel agent feature extraction data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_evaluation_data(data: list) -> tuple:
    """
    Prepare data for evaluation by converting features lists to sets
    for case-insensitive matching evaluation
    """
    true_features = []
    communication_features = []
    productivity_features = []
    technical_features = []
    final_features = []
    
    for review in data:
        # Convert feature lists to sets for case-insensitive matching
        true_set = set(feature.lower() for feature in review['output'])
        communication_set = set(feature.lower() for feature in review['communication_features'])
        productivity_set = set(feature.lower() for feature in review['productivity_features'])
        technical_set = set(feature.lower() for feature in review['technical_features'])
        final_set = set(feature.lower() for feature in review['final_features'])
        
        true_features.append(true_set)
        communication_features.append(communication_set)
        productivity_features.append(productivity_set)
        technical_features.append(technical_set)
        final_features.append(final_set)
    
    return true_features, communication_features, productivity_features, technical_features, final_features

def calculate_metrics(true_features: list, predicted_features: list) -> dict:
    """
    Calculate precision, recall, and F1 score for feature extraction.
    """
    # Convert sets to lists for sklearn metrics
    true_list = [list(features) for features in true_features]
    pred_list = [list(features) for features in predicted_features]
    
    # Create MultiLabelBinarizer and fit it on all features
    mlb = MultiLabelBinarizer()
    all_features = set()
    for features in true_list + pred_list:
        all_features.update(features)
    mlb.fit([list(all_features)])
    
    # Transform the feature lists to binary arrays
    true_binary = mlb.transform(true_list)
    pred_binary = mlb.transform(pred_list)
    
    # Calculate metrics
    precision = precision_score(true_binary, pred_binary, average='micro', zero_division=0)
    recall = recall_score(true_binary, pred_binary, average='micro', zero_division=0)
    f1 = f1_score(true_binary, pred_binary, average='micro', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_parallel_agent_results(file_path: str) -> dict:
    """
    Evaluate the parallel agent feature extraction results
    """
    # Load data
    data = load_parallel_agent_data(file_path)
    
    # Prepare data for evaluation
    true_features, communication_features, productivity_features, technical_features, final_features = prepare_evaluation_data(data)
    
    # Calculate metrics for all approaches
    communication_metrics = calculate_metrics(true_features, communication_features)
    productivity_metrics = calculate_metrics(true_features, productivity_features)
    technical_metrics = calculate_metrics(true_features, technical_features)
    final_metrics = calculate_metrics(true_features, final_features)
    
    # Combine results
    results = {
        'communication_metrics': communication_metrics,
        'productivity_metrics': productivity_metrics,
        'technical_metrics': technical_metrics,
        'final_metrics': final_metrics,
        'total_reviews': len(data),
        'total_true_features': sum(len(features) for features in true_features),
        'total_communication_features': sum(len(features) for features in communication_features),
        'total_productivity_features': sum(len(features) for features in productivity_features),
        'total_technical_features': sum(len(features) for features in technical_features),
        'total_final_features': sum(len(features) for features in final_features)
    }
    
    return results

def save_parallel_agent_evaluation_results(file_path: str, output_dir: str, results: dict):
    """Save parallel agent evaluation results to a text file"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    base_filename = os.path.basename(file_path).replace('.json', '')
    output_filename = f"{base_filename}-evaluation.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    # Prepare output text
    output_lines = []
    output_lines.append("Parallel Agent Feature Extraction Evaluation Results")
    output_lines.append("=" * 70)
    output_lines.append(f"Total Reviews: {results['total_reviews']}")
    output_lines.append(f"Total True Features: {results['total_true_features']}")
    output_lines.append(f"Total Communication Features: {results['total_communication_features']}")
    output_lines.append(f"Total Productivity Features: {results['total_productivity_features']}")
    output_lines.append(f"Total Technical Features: {results['total_technical_features']}")
    output_lines.append(f"Total Final Features: {results['total_final_features']}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("COMMUNICATION AGENT RESULTS")
    output_lines.append("-" * 40)
    output_lines.append(f"Precision: {results['communication_metrics']['precision']:.4f}")
    output_lines.append(f"Recall: {results['communication_metrics']['recall']:.4f}")
    output_lines.append(f"F1 Score: {results['communication_metrics']['f1']:.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("PRODUCTIVITY AGENT RESULTS")
    output_lines.append("-" * 40)
    output_lines.append(f"Precision: {results['productivity_metrics']['precision']:.4f}")
    output_lines.append(f"Recall: {results['productivity_metrics']['recall']:.4f}")
    output_lines.append(f"F1 Score: {results['productivity_metrics']['f1']:.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("TECHNICAL AGENT RESULTS")
    output_lines.append("-" * 40)
    output_lines.append(f"Precision: {results['technical_metrics']['precision']:.4f}")
    output_lines.append(f"Recall: {results['technical_metrics']['recall']:.4f}")
    output_lines.append(f"F1 Score: {results['technical_metrics']['f1']:.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("COORDINATOR AGENT (FINAL) RESULTS")
    output_lines.append("-" * 40)
    output_lines.append(f"Precision: {results['final_metrics']['precision']:.4f}")
    output_lines.append(f"Recall: {results['final_metrics']['recall']:.4f}")
    output_lines.append(f"F1 Score: {results['final_metrics']['f1']:.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("PERFORMANCE COMPARISON")
    output_lines.append("-" * 40)
    
    # Find best individual agent
    agent_scores = [
        ("Communication", results['communication_metrics']['f1']),
        ("Productivity", results['productivity_metrics']['f1']),
        ("Technical", results['technical_metrics']['f1'])
    ]
    best_agent = max(agent_scores, key=lambda x: x[1])
    
    output_lines.append(f"Best Individual Agent: {best_agent[0]} (F1: {best_agent[1]:.4f})")
    output_lines.append(f"Coordinator Agent F1: {results['final_metrics']['f1']:.4f}")
    
    coordinator_improvement = results['final_metrics']['f1'] - best_agent[1]
    output_lines.append(f"Coordinator vs Best Agent: {coordinator_improvement:+.4f}")
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("DETAILED METRICS COMPARISON")
    output_lines.append("-" * 40)
    output_lines.append("Agent          | Precision | Recall   | F1 Score")
    output_lines.append("-" * 70)
    output_lines.append(f"Communication  | {results['communication_metrics']['precision']:.4f}     | {results['communication_metrics']['recall']:.4f}     | {results['communication_metrics']['f1']:.4f}")
    output_lines.append(f"Productivity   | {results['productivity_metrics']['precision']:.4f}     | {results['productivity_metrics']['recall']:.4f}     | {results['productivity_metrics']['f1']:.4f}")
    output_lines.append(f"Technical      | {results['technical_metrics']['precision']:.4f}     | {results['technical_metrics']['recall']:.4f}     | {results['technical_metrics']['f1']:.4f}")
    output_lines.append(f"Coordinator    | {results['final_metrics']['precision']:.4f}     | {results['final_metrics']['recall']:.4f}     | {results['final_metrics']['f1']:.4f}")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Parallel agent evaluation results saved to: {output_path}")
    return output_path

def main():
    # Get configuration from command line arguments
    config = get_configuration()
    input_file = config['INPUT_FILE']
    output_dir = config['OUTPUT_DIR']
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        print("Please run the parallel agent feature extraction first.")
        return
    
    # Evaluate results
    results = evaluate_parallel_agent_results(input_file)
    
    # Save results
    save_parallel_agent_evaluation_results(input_file, output_dir, results)
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("PARALLEL AGENT EVALUATION SUMMARY")
    print("=" * 70)
    
    # Find best individual agent
    agent_scores = [
        ("Communication", results['communication_metrics']['f1']),
        ("Productivity", results['productivity_metrics']['f1']),
        ("Technical", results['technical_metrics']['f1'])
    ]
    best_agent = max(agent_scores, key=lambda x: x[1])
    
    print(f"Best Individual Agent: {best_agent[0]} (F1: {best_agent[1]:.4f})")
    print(f"Coordinator Agent F1: {results['final_metrics']['f1']:.4f}")
    
    coordinator_improvement = results['final_metrics']['f1'] - best_agent[1]
    print(f"Coordinator vs Best Agent: {coordinator_improvement:+.4f}")

if __name__ == '__main__':
    main()
