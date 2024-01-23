import os
import fire
import torch
import pandas as pd
from tqdm import tqdm

from eval_utils import load_model, generate_text

def main(
    model_name,
    output_path: str,
    peft_model: str = None,
    quantization: bool = False,
    max_new_tokens=10,  # The maximum numbers of tokens to generate
    prompt_file: str = None,  # CSV file with prompts in the user_prompt column
    seed: int = 42,  # seed value for reproducibility
    do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool = False,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float = 1.0,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
    top_k: int = 10,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int = 1,  # [optional] Exponential penalty to the length that is used with beam-based generation.
    # enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
    # enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
    # enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
    max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False,  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs,
):
    print(quantization)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model, tokenizer = load_model(
        model_name, peft_model, quantization, use_fast_kernels
    )

    # Load in prompts to run from a csv file
    prompts_df = pd.read_csv(prompt_file)

    for i in range(5):
        model_outputs = list()
        for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
            user_prompt = row["user_prompt"]
            # question = row["question"]
            correct_answer = row["correct_answer"]
            try:
                output_text = generate_text(
                    model,
                    tokenizer,
                    user_prompt,
                    max_padding_length,
                    max_new_tokens,
                    do_sample,
                    top_p,
                    temperature,
                    min_length,
                    use_cache,
                    top_k,
                    repetition_penalty,
                    length_penalty,
                    **kwargs,
                )
                model_response = output_text.split(user_prompt)[-1].strip()
            except:
                print("Error during generation")
                model_response = "No Output"

            # print(model_response, correct_answer)
            model_outputs.append(
                {
                    "user_prompt": user_prompt,
                    f"model_response_{i}": model_response,
                }
            )

        model_outputs = pd.DataFrame(model_outputs)
        model_outputs.to_csv(os.path.join(output_path, f"results_v{i}.csv"))

        prompts_df = prompts_df.merge(model_outputs, on="user_prompt")

    prompts_df.to_csv(os.path.join(output_path, f"combined_results.csv"))


if __name__ == "__main__":
    fire.Fire(main)
