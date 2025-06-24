# Paper-MuC
Code and implementation details for paper MuC: A Multi-Core Tucker Model with Core Attention

## Hyperparameter Settings

The table lists the best-performing hyperparameter configurations of MuC on each dataset.

| Hyperparameter        | FB15k-237 | YAGO3-10 | WN18RR |
|-----------------------|-----------|----------|--------|
| Embedding dimension D | 200       | 200      | 200    |
| Core tensor dimension P | 100     | 100      | 100    |
| Number of cores C     | 5         | 5        | 3      |
| Learning rate         | 0.1       | 0.1      | 0.1    |
| Batch size            | 256       | 256      | 256    |
| λ₁                    | 0.05      | 0.03     | 0.03   |
| λ₂                    | 0.05      | 0.01     | 0.005  |
| λ₃                    | 0.005     | 0.003    | 0.0005 |

