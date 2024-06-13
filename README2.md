
## UniGen Generation




## UniGen Augmentation Using LLaMA-Factory

### Requirements

- [`LLaMA-Factory`](https://github.com/hiyouga/LLaMA-Factory) installed

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

### Fine-Tuning 
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
