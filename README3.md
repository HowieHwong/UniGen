To assess the performance of LLMs on the generated dataset, follow these steps:

1. **Generate Output:** First, generate the output using the provided dataset.
2. **Evaluate the Output:** After generating the output, proceed to evaluate its performance.

To generate the output for the LLMs on your dataset, execute the following command:

```shell
unigen-cli evaluation unigen/examples/eval_config.yaml
```

You can customize your evaluation settings in the configuration file located at: `unigen/examples/eval_config.yaml`

Specify the file to be evaluated in the `tasks_files` section of your configuration. Include the filename and the temperature setting for the evaluation process.

