# Research goal

## Mandatory training constraint

**Never use gradient descent anywhere in this project.** This applies to every parameter, including output weights, biases and piecewise endpoints, and to every backend, experiment, baseline and diagnostic. Using gradient descent anywhere is a fatal project error. Remove any implementation, checkpoint, or published result that depends on it. Check the update equation itself. Multiplying an output feature by the numerical residual to fit an output weight is squared-error gradient descent even when implemented by hand or called Hebbian. This prohibition includes backpropagation, automatic differentiation and gradient-based optimizers.

Use parameter-free output pooling. Update hidden weights only from local presynaptic activity, postsynaptic activity and the signed class teaching signal. The final residual supplies only its sign to training. Error magnitudes may be reported for evaluation and checkpoint selection.

For the manuscript's sine, Blue Marble and CelebA experiments, use a single Lion update for each run, explicit or semi-implicit, from initialization through the final minibatch. Keep the corresponding coactivity history whenever training continues from a checkpoint within the run.

Use the 256-value bottleneck for the manuscript's CelebA model and its Lion comparison.

Use one cosine step-size schedule over the full training run. A short warmup is optional. For the coordinate demonstrations, use one starting step shared by all layers and keep the batch size constant. Run the schedule continuously from initialization. Select checkpoints for reporting without restarting training from the best intermediate checkpoint. The sine and Blue Marble examples should show what a standard MLP learns with a simple training procedure.

## Architecture

Develop a generic, fully connected mixed-class MLP that callers can use as a black-box regressor. One shared set of weights must serve the entire input domain. The same architecture and learning rule should work with ordinary input and target vectors across regression tasks.

- Train every hidden layer with local Hebbian updates throughout training. Use the final output residual as the teaching signal for the signed classes. Readout-only training with frozen hidden features is out of scope, including as a diagnostic or baseline. Backpropagation and offline linear solves are also out of scope.
- Keep connections dense across classes in adjacent layers. Per-class or per-layer winner selection within a shared MLP is in scope. It selects activations after a shared layer has evaluated its weights.
- Exclude spatial routing, grid-based sites, region-specific subnetworks or parameter banks, coordinate lookup, and features or partitions designed around a benchmark's known structure.
- Explore capacity directly by varying hidden width and depth. A prediction from each hidden layer may contribute to one combined output, with the residual of that output teaching every hidden layer.
- Treat sine, staircase, and image fitting as diagnostic benchmarks. A proposed architectural or learning-rule change should be usable on a new regression task given only its input and target vectors. Keep benchmark-specific knowledge in evaluation. Judge progress by what the shared MLP learns and report misses and limitations directly.

## Scientific writing

Describe each algorithm step by naming the quantity, the operation and its result. For example, “the increase class receives $e_k$ and the decrease class receives $-e_k$” specifies both teaching signals. Define each symbol at first use. Distinguish one example's contribution to signed coactivity, the minibatch average and the resulting weight update.

Report each measurement in terms of what was counted or measured. A layer's share of the parameters supports a statement about model size. A claim about computational cost needs an operation count or timing.

Use “image” for autoencoder inputs, reconstructions and generated samples in manuscript text, figures and documentation. Use “CelebA image” when the dataset matters.
