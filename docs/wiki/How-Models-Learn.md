# How does a model learn?

Imagine learning from practice questions that already include the answers. You make a guess, compare it with the answer, and adjust how you approach the next question. **Supervised learning** follows that pattern using numerical calculations.

![Diagram: How Models Learn](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/learning-loop.png)

This loop happens on **training rows**. Later, we evaluate the model on separate rows. The goal is to learn useful patterns, rather than just fit the practice answers.

## The models we compared

| Model | Beginner-friendly picture | Why include it? |
|---|---|---|
| Majority-class dummy | Always answers “No Diabetes” | Shows how high accuracy can be achieved without identifying any minority cases |
| Logistic regression | A learned scoring formula combining the inputs | A simple, interpretable baseline |
| MLP neural network | Several layers transform inputs into increasingly useful internal representations | Can learn nonlinear relationships among indicators |
| Boosted trees | Many small decision trees are built in sequence to improve the prediction objective | Another way to learn nonlinear patterns; we tested histogram boosting and CatBoost |
| Soft-voting ensemble | Several models contribute scores, which are averaged | Tests whether combining models improves performance |

More complexity did not consistently give better results.

## Inside the Experiment 1 neural network

![The Experiment 1 weighted neural network has 21 inputs, hidden layers with 64 and 32 units, and three class scores.](https://raw.githubusercontent.com/wiki/NewOsakA/diabetes-ml/assets/neural-network.png)

- **21 inputs:** the survey indicators.
- **64 and 32 hidden units:** adjustable calculations inside the network. These are not additional observed health measurements.
- **Three output scores:** one for each class.
- **ReLU:** an activation function that lets the network learn nonlinear patterns.

We first **standardize** inputs for logistic regression and MLPs. This puts inputs with different numeric scales on a more comparable scale. The scaling settings are learned from training rows only. Trees receive the numeric survey codes directly.

## Five words you will see in the training logs

| Word | Meaning here |
|---|---|
| Batch | A small group of examples used for an update; our neural batches contain 256 rows |
| Epoch | One training pass through the examples, or the configured number of sampled examples |
| Loss | A numeric penalty that training tries to reduce |
| Backpropagation | The calculation of how network parameters contribute to that loss |
| AdamW | The optimizer used to update neural-network parameters |

The Experiment 1 neural networks trained for 30 epochs. The selected weighted MLP's saved checkpoint was from **epoch 20**, because that checkpoint had the best tuning macro F1. The last epoch is not automatically the best.

## Three scores do not automatically mean three reliable probabilities

A model must also choose a final label. Ordinary **argmax** chooses the class with the largest score. **Decision tuning** can change that choice to reflect a different evaluation goal.

Scores from weighted or focal-loss training may not be calibrated. For example, a score of 0.8 does not automatically mean an 80% observed frequency. [Training Techniques](Training-Techniques) explains calibration separately from label decisions.

**Next:** [The training techniques we tested →](Training-Techniques)
