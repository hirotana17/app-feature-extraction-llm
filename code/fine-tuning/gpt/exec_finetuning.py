import argparse
from openai import OpenAI
from dotenv import load_dotenv

def main():
  parser = argparse.ArgumentParser(description='Fine-tuning execution script')
  parser.add_argument('--file', '-f', required=True, help='Path to the training data file')
  parser.add_argument('--model', '-m', required=True, help='Model name for fine-tuning')

  args = parser.parse_args()

  load_dotenv()
  client = OpenAI()

  # Generate a training file
  training_file = client.files.create(
      file=open(args.file, "rb"),
      purpose="fine-tune"
    )
  training_file_id = training_file.id

  # Fine-tuning job
  client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model=args.model,
    # method= {
    #   "type": "supervised",
    #   "supervised": {
    #     "hyperparameters": {
    #       "batch_size": "2",
    #       "learning_rate_multiplier": "1.8",
    #       "n_epochs": "3",
    #     }
    #   }
    # },
  )

if __name__ == "__main__":
  main()

  # "./data/in-domain/bin0/fine_tuning_data/train-set-finetuning-promptft1.json"
  # "gpt-4.1-nano-2025-04-14" 

