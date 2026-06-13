import json
import os
import re
import sys
import argparse
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import prompt

# Load environment variables
load_dotenv()

def get_configuration():
    """Parse command line arguments and build configuration"""
    parser = argparse.ArgumentParser(description='Feature extraction script')
    parser.add_argument('-d', '--directory', required=True,
                       help='Base directory path (e.g., ./data/in-domain/bin0/)')
    parser.add_argument('-m', '--model', required=True,
                       help='Model name for feature extraction')
    
    args = parser.parse_args()
    
    # Build configuration from the two main options
    base_dir = args.directory.rstrip('/')  # Remove trailing slash if present
    
    config = {
        'INPUT_DIR': f'{base_dir}/formatted_original_data',
        'INPUT_FILE': 'test-set.json',
        'OUTPUT_DIR': f'{base_dir}/feature_extracted_data',
        'MODEL_NAME': args.model,
        'SYSTEM_PROMPT': prompt.PROMPTFT1,
        'PROMPT_NAME': 'promptft1'
    }
    
    return config

class FeatureExtraction(BaseModel):
    """Schema for feature extraction output"""
    features: List[str] = Field(description="List of feature names extracted from the review")

def extract_features(review_text, config, review_number=None, ground_truth_features=None):
    """Extract features from a review text"""
    # Initialize LLM with structured output for deterministic parsing
    llm = ChatOpenAI(model_name=config['MODEL_NAME'], temperature=0).with_structured_output(FeatureExtraction)
    
    # Define prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", config['SYSTEM_PROMPT']),
        ("human", "Review text: {review_text}")
    ])
    
    try:
        response: FeatureExtraction = llm.invoke(prompt_template.format_messages(review_text=review_text))
        features = [feature.strip() for feature in response.features if feature.strip()]
        
        # Display results
        review_num_str = f"[{review_number}] " if review_number is not None else ""
        ground_truth_str = f" | GT: {ground_truth_features}" if ground_truth_features else ""
        print(f"{review_num_str}Review: {review_text[:100]}... | Extracted: {features}{ground_truth_str}")
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return []

def process_reviews(input_file, output_file, config):
    """Process reviews and extract features"""
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    print(f"\nProcessing {len(reviews)} reviews...")
    
    # Process each review
    for i, review in enumerate(reviews, 1):
        # Add review number
        review['review_number'] = i
        
        # Extract features
        ground_truth = review.get('output', [])
        extracted_features = extract_features(
            review['input'],
            config,
            review_number=i,
            ground_truth_features=ground_truth
        )
        
        # Add extracted features to the review
        review['extracted_features'] = extracted_features
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted. Results saved to {output_file}")

def main():
    # Get configuration from command line arguments
    config = get_configuration()
    
    # Generate output filename with readable model label.
    raw_model_name = config['MODEL_NAME']
    if raw_model_name.startswith("ft:"):
        # Example: ft:gpt-4.1-nano-2025-04-14:org::id -> gpt-4.1-nano-2025-04-14
        parts = raw_model_name.split(":")
        model_label_source = parts[1] if len(parts) > 1 else raw_model_name
    else:
        model_label_source = raw_model_name.split("/")[-1]
    model_label = re.sub(r"[^a-z0-9]+", "-", model_label_source.lower()).strip("-") or "model"
    input_file = os.path.join(config['INPUT_DIR'], config['INPUT_FILE'])
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    base_name = os.path.basename(input_file).replace('.json', '')
    output_file = os.path.join(config['OUTPUT_DIR'], f"{base_name}-{model_label}-singleagent.json")
    
    # Execute review processing
    process_reviews(input_file, output_file, config)

if __name__ == '__main__':
    main()
