<p align="center">
  <img src="https://github.com/HowieHwong/UniGen/blob/main/images/unigen_logo.png" alt="logo">
</p>


# UniGen: A Unified Framework for Textual Dataset Generation

<div align="center">

<div align="center">

<a href="https://arxiv.org/">
    <img src="https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightgrey?style=flat-square" alt="Paper" style="height:30px;"/>
</a>
<a href="https://unigen-framework.github.io/">
    <img src="https://img.shields.io/badge/Website-%F0%9F%90%BE-green?style=flat-square" alt="Website" style="height:30px;"/>
</a>


</div>

</div>

- [UniGen: A Unified Framework for Textual Dataset Generation](#unigen-a-unified-framework-for-textual-dataset-generation)
  - [What's UniGen?](#gem-whats-unigen)
  - [Features](#dart-features)
- [Usage](#closed_book-usage)
  - [UniGen Dataset Generation](#unigen-dataset-generation)
    - [Configuration Settings](#configuration-settings)
      - [API Settings](#api-settings)
      - [Generation Settings](#generation-settings)
      - [Generation Hint](#generation-hint)
      - [Efficiency Configuration](#efficiency-configuration)
    - [Step-by-Step Guide](#step-by-step-guide)
  - [LLM Evaluation](#llm-evaluation)
    - [Generate Output](#1-generate-output)
    - [Evaluate the Output](#2-evaluate-the-output)
  - [UniGen Augmentation Using LLaMA-Factory](#unigen-augmentation-using-llama-factory)
    - [Requirements](#requirements)
    - [Fine-Tuning](#fine-tuning)


## :gem: What's UniGen?

Large Language Models (LLMs) such as GPT-4 and Llama3 have significantly impacted various fields by enabling high-quality synthetic data generation and reducing dependence on expensive human-generated datasets. 
Despite this, challenges remain in the areas of generalization, controllability, diversity, and truthfulness within the existing generative frameworks. To address these challenges, this paper presents UniGen, a comprehensive LLM-powered framework designed to produce diverse, accurate, and highly controllable datasets. UniGen is adaptable, supporting all types of text datasets and enhancing the generative process through innovative mechanisms. To augment data diversity, UniGen incorporates an attribute-guided generation module and a group checking feature. For accuracy, it employs a code-based mathematical assessment for label verification alongside a retrieval-augmented generation technique for factual validation. The framework also allows for user-specified constraints, enabling customization of the data generation process to suit particular requirements. 

<span style="color: red;">TL;DR: UniGen is an LLM-powered framework designed to generate diverse, accurate, and highly controllable text datasets.</span>

![UniGen Architecture](https://github.com/HowieHwong/UniGen/blob/main/images/architecture.png "UniGen Framework Architecture")


## :dart: Features

- Generalization: UniGen supports all textual datasets as input to generate a new dataset.
- Diversity: We support `Attribute-Guided Generation`, `Diverse Example Selection for ICL`, and `Group Checking` to enhance data diversity.
- Truthfulness: `Self-Evaluation`, `Code-Based Validation`, and `RAG-Based Validation` are equipped to ensure truthfulness.
- Controllability: UniGen accepts user `constraints` to make generation more controllable.
- Various Application: UniGen can be applied for `dynamic benchmark` or `data augmentation`.







## :closed_book: Usage

### Installation
```bash
git clone --depth 1 git@github.com:HowieHwong/UniGen.git
cd UniGen
pip install -e .
```

### UniGen Dataset Generation
This guide provides detailed instructions for generating a dataset using the UniGen tool with the specified configuration settings.

#### Configuration Settings

Below are the configurations to be used for data generation:

##### API Settings

```yaml
api_settings:
  model_type: "gpt"
  api_key: "YOUR_API_KEY_HERE"
  embedding_model: "text-embedding-3-large"
  use_azure: false
  api_base: null
  azure_version: null
```

- `model_type`: Specifies the type of model to use (e.g., GPT).
- `api_key`: Your API key for accessing the model.
- `embedding_model`: The embedding model to use.
- `use_azure`: Boolean to indicate if Azure should be used.
- `api_base`: Base URL for the API (if using Azure).
- `azure_version`: Version of Azure to use (if applicable).

##### Generation Settings

```yaml
generation_settings:
  temperature: 0.8
  generation_number: 100
  batch_size: 5
  few_shot_num: 5
  random_example: false
  max_worker: 2
```

- `temperature`: Controls the randomness of the generation.
- `generation_number`: Total number of items to generate.
- `batch_size`: Number of items to generate in each batch.
- `few_shot_num`: Number of examples to use for few-shot learning.
- `random_example`: Boolean to indicate if random examples should be used.
- `max_worker`: Number of worker threads to use for generation.

##### Generation Hint

```yaml
generation_hint:
  dataset_description: "It is a dataset of high quality linguistically diverse grade school math word problems created by human problem writers. These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer. A bright middle school student should be able to solve every problem. It can be used for multi-step mathematical reasoning. Each problem should only have one question and one correct answer."
  dataset_name: "GSM8K"
  original_dataset_path: "path/to/original/dataset"
  dataset_constraint: []
  with_label: true
  with_attribute: false
  add_attribute: false
  extra_info_keys: []
```

- `dataset_description`: A detailed description of the dataset.
- `dataset_name`: The name of the dataset.
- `original_dataset_path`: Path to the original dataset.
- `dataset_constraint`: List of constraints for generating the dataset.
- `with_label`: Boolean to indicate if labels should be included.
- `with_attribute`: Boolean to indicate if attributes should be included.
- `add_attribute`: Boolean to indicate if new attributes should be added.
- `extra_info_keys`: List of extra information keys.

##### Efficiency Configuration

```yaml
efficiency_configuration:
  learn_from_mistake: false
  learn_from_human_feedback: false
  feedback_iteration: 1
  self_reflection: true
  math_eval: true
  truthfulness_eval: false
```

- `learn_from_mistake`: Boolean to indicate if the model should learn from mistakes.
- `learn_from_human_feedback`: Boolean to indicate if the model should learn from human feedback.
- `feedback_iteration`: Number of feedback iterations.
- `self_reflection`: Boolean to indicate if self-reflection should be used.
- `math_eval`: Boolean to indicate if mathematical evaluation should be used.
- `truthfulness_eval`: Boolean to indicate if truthfulness evaluation should be used.

#### Step-by-Step Guide

##### 1. Set Up API Key

Replace `"YOUR_API_KEY_HERE"` in the `api_settings` with your actual API key.

##### 2. Prepare the Original Dataset

Ensure that the original dataset is available at the specified path in `original_dataset_path`.

##### 3. Configure the Generation Settings

Adjust the `generation_settings` according to your specific needs. Ensure the values align with your data generation goals.

##### 4. Define Generation Hints

Provide detailed information in `generation_hint` to guide the data generation process. This helps in creating high-quality and relevant data.

##### 5. Efficiency Configuration

Configure efficiency settings to optimize the data generation process. Enable or disable features like learning from mistakes and self-reflection based on your requirements.

##### 6. Run Data Generation

Use the configured settings to generate the dataset using the UniGen tool. Ensure all configurations are correctly set before starting the generation process.

```shell
unigen-cli gene examples/eval_generation.yaml
```

### LLM Evaluation

To assess the performance of LLMs on the generated dataset, follow these steps:

#### 1. Generate Output

First, generate the output using the provided dataset. Execute the following command to generate the output for the LLMs on your dataset:

```shell
unigen-cli evaluation examples/eval_config.yaml
```

You can customize your evaluation settings in the configuration file located at `examples/eval_generation_config.yaml`.

#### 2. Evaluate the Output

After generating the output, proceed to evaluate its performance. Use the following command to evaluate the generated output:

```shell
unigen-cli judge examples/eval_config.yaml
```

Customize your evaluation settings in the configuration file located at `examples/eval_judge.yaml`.

Specify the file to be evaluated in the `tasks_files` section of your configuration. Include the filename and the temperature setting for the LLM-judge process.

### UniGen Augmentation Using LLaMA-Factory

This section aims to introduce how to use the generated data powered by UniGen to conduct data augmentation on your LLMs.

#### Requirements

- [`LLaMA-Factory`](https://github.com/hiyouga/LLaMA-Factory) installed

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

#### Fine-Tuning 
To use LLaMA-Factory, follow these steps:

1. Add your dataset into `dataset_info.json`.

   - The `dataset_info.json` file should include details about your dataset. For more information, refer to the [LLaMA-Factory data documentation](https://github.com/hiyouga/LLaMA-Factory/tree/main/data).

2. Update the `train_config.yaml` file with your dataset information and training parameters.

3. Start the training process with the following command:

   ```bash
   llamafactory-cli train train_config.yaml
   ```

4. Once training is complete, you can run model inference using the command:

   ```bash
   llamafactory-cli api model_inference.yaml
   ```

# Cite UniGen

```text

```