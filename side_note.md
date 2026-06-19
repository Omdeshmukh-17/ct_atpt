# Side Note: Understanding CT-ATPT abg Values & Token Pruning

## What are the `abg` values?
In the CT-ATPT training logs, `abg=[alpha, beta, gamma]` (e.g., `abg=[0.333, 0.333, 0.333]`) represents the softmax-normalized importance weights for the three token pruning metrics.

These are defined as parameters in the [CTATPT](file:///c:/Users/Aries/Documents/Codex/2026-05-15/research/ct_atpt/model.py#L119) model via `self.importance_logits` and are learned dynamically via backpropagation:

$$\text{Weights} = \text{Softmax}(\text{importance\_logits}) = [\alpha, \beta, \gamma]$$

1. **Alpha ($\alpha$, `abg[0]`) — Attention Received:** How much attention a patch token receives from other tokens in the layer.
2. **Beta ($\beta$, `abg[1]`) — Attention Rollout:** The cumulative flow of attention from the `[CLS]` token down to the patch tokens, showing which global representations the classifier is relying on.
3. **Gamma ($\gamma$, `abg[2]`) — Patch Energy:** Local high-frequency details (intensity variance and spatial gradients) from the original 3D volume patches.

---

## Does `abg=[0.333, 0.333, 0.333]` mean the model is just a standard ViT?
**No.** Even when the importance weights are equal, the model is fundamentally different from a standard ViT:

### 1. Warmup vs. Active Pruning
* **During Warmup (`epoch < pruning_warmup_epochs`):** Pruning is completely disabled (`keep=[1.0, 1.0, ...]`) to allow the backbone to learn stable features. During this time, the weights default to `[0.333, 0.333, 0.333]` and the model acts as a standard ViT.
* **During Active Pruning:** Once warmup ends, the model begins pruning tokens based on the joint score. Even with uniform weights, the bottom tokens are discarded, which a standard ViT never does.

### 2. Key Differences in Active Pruning Mode
* **Token Discarding:** Standard ViTs pass 100% of the tokens through all layers. CT-ATPT dynamically drops uninformative tokens layer-by-layer.
* **Information Recycling:** Before tokens are pruned, their features are recycled into the `[CLS]` token using a softmax-weighted sum (see [model.py:L279-288](file:///c:/Users/Aries/Documents/Codex/2026-05-15/research/ct_atpt/model.py#L279-L288)).
* **Learning the Weights:** Over epochs, gradients flow back into `importance_logits`, dynamically shifting `abg` away from `[0.333, 0.333, 0.333]` to favor whichever indicators are most predictive for classification.
