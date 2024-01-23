import os, pickle, json, time, fire, logging, transformers

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llama2_test")

transformers.logging.set_verbosity_info()

logger.info(os.getcwd())
os.chdir("/home/ubuntu/llm_testing/")
logger.info(os.getcwd())


def prepare_prompts(base_prompt_path: str, prompts_path: str):
    """Function that adds base prompts (for Chain of thought or few shot prompting) to each prompt for inference

    Params:
        base_prompt_path (str): path to a pickle file containing the base prompt
        prompts_path (str): path to a pickle file containing the prompts to run on the llm

    Returns:
        prompts (list str): list of prompts to run with the base prompts prepended to each
        pmt_raw (list str): list of prompts as is
        answers (list tuple): list of expected answer tuple (option, answer)
    """

    # Data prep
    with open(base_prompt_path, "rb") as f:
        base_prompt = pickle.load(f)
    with open(prompts_path, "r") as f:
        new_prompts = json.load(f)

    prompts = []
    pmt_raw = []
    answers = []

    # Prepare prompts
    for pmt in new_prompts:
        pmt_fmt = f"""{base_prompt}
        
    {pmt['question']}
    Answer: """

        prompts.append(pmt_fmt)
        pmt_raw.append(pmt)
        answers.append((pmt["answer_letter"], pmt["answer"]))

    return prompts, pmt_raw, answers


def load_llm_pipeline(model_path: str):
    """Function to load the LLaMa model and create pipeline

    Params:
        model_path (str): path to the model storage directory

    Returns:
        pipeline (transformers.Pipeline): Pipeline
        tokenizer (transformers.LlamaTokenizer): LlamaTokennizer
    """
    logger.info("Loading Llama2 70B Chat")
    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_path, legacy=False)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        device_map="auto",
    )
    return pipeline, tokenizer


def main(
    model_path: str = "../llama-2-70b-chat-hf",
    output_dir: str = "results/",
    base_prompt_path: str = "../cot_base.pkl",
    prompts_path: str = "../training_prompts.json",
):
    """Function to run text generation on prompts"""
    logger.info("Loading base prompt and prompts to run")

    prompts, pmt_raw, answers = prepare_prompts(base_prompt_path, prompts_path)

    start_time = time.time()
    pipeline, tokenizer = load_llm_pipeline(model_path)
    model_prep_time = time.time()

    logger.info("Finished model loading")
    logger.info("Model prep time: {}".format(model_prep_time - start_time))

    sequences = pipeline(
        prompts,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=10,
    )

    inf_time = time.time()
    logger.info("Finished running inference")
    logger.info("Total inference time: {}".format(inf_time - model_prep_time))

    time_analysis = {"model_prep_time": model_prep_time, "inf_time": inf_time}

    # Saving processed prompts and outputs
    with open(os.path.join(output_dir, "pmt_raw.pkl"), "wb") as f:
        pickle.dump(pmt_raw, f)
    with open(os.path.join(output_dir, "answers.pkl"), "wb") as f:
        pickle.dump(answers, f)
    with open(os.path.join(output_dir, "outputs.pkl"), "wb") as f:
        pickle.dump(sequences, f)
    with open(os.path.join(output_dir, "time_analysis.pkl"), "wb") as f:
        pickle.dump(time_analysis, f)


if __name__ == "__main__":
    fire.Fire(main)
