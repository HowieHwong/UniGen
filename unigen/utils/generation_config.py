deepinfra_model = [
    "llama2-70b",
    "llama2-13b",
    "llama2-7b",
    "mistral-7b",
    "dolly-12b",
    "mixtral-8x7B",
    "yi-34b"]
zhipu_model = ["glm-4", "glm-3-turbo"]
claude_model = ["claude-2", "claude-instant-1"]
openai_model = ["chatgpt", "gpt-4"]
google_model = ["bison-001", "gemini"]
wenxin_model = ["ernie"]
replicate_model=["vicuna-7b","vicuna-13b","vicuna-33b","chatglm3-6b","llama3-70b","llama3-8b"]

online_model = deepinfra_model + zhipu_model + claude_model + openai_model + google_model + wenxin_model+replicate_model

model_info = {
    "online_model": online_model,
    "zhipu_model": zhipu_model,
    "deepinfra_model": deepinfra_model,
    'claude_model': claude_model,
    'openai_model': openai_model,
    'google_model': google_model,
    'wenxin_model': wenxin_model,
    'replicate_model':replicate_model,
    "model_mapping": {
        "baichuan-inc/Baichuan-13B-Chat": "baichuan-13b",
        "baichuan-inc/Baichuan2-13B-chat": "baichuan2-13b",
        "01-ai/Yi-34B-Chat": "yi-34b",
        "THUDM/chatglm2-6b": "chatglm2",
        "THUDM/chatglm3-6b": "chatglm3",
        "lmsys/vicuna-13b-v1.3": "vicuna-13b",
        "lmsys/vicuna-7b-v1.3": "vicuna-7b",
        "lmsys/vicuna-33b-v1.3": "vicuna-33b",
        "meta-llama/Llama-2-7b-chat-hf": "llama2-7b",
        "meta-llama/Llama-2-13b-chat-hf": "llama2-13b",
        "meta/meta-llama-3-70b-instruct":"llama3-70b",
        "meta/meta-llama-3-8b-instruct":"llama3-8b",
        "TheBloke/koala-13B-HF": "koala-13b",
        "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": "oasst-12b",
        "WizardLM/WizardLM-13B-V1.2": "wizardlm-13b",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "mixtral-8x7B",
        "meta-llama/Llama-2-70b-chat-hf": "llama2-70b",
        "mistralai/Mistral-7B-Instruct-v0.1": "mistral-7b",
        "databricks/dolly-v2-12b": "dolly-12b",
        "bison-001": "bison-001",
        "ernie": "ernie",
        "chatgpt": "chatgpt",
        "gpt-4": "gpt-4",
        "claude-2": "claude-2",
        "glm-4": "glm-4",
        "glm-3-turbo": "glm-3-turbo"
    }
}