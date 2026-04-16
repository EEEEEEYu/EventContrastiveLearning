### **Title: Orthogonal Disentanglement for Robust Event-RGB Representation Learning in Visual Object Tracking**

**1\. Motivation and Related Work**

While the physical complementarity of RGB and Event cameras is well established—RGB capturing dense spatial texture, Event capturing high-frequency asynchronous temporal dynamics—current multimodal fusion architectures fundamentally fail to preserve this separation under stress. Standard models entangle these modalities into a shared latent space, meaning localized sensor failure (e.g., severe RGB motion blur or Event sensor thermal flicker) poisons the entire representation.

Recent literature, notably the 2025 "RED" architecture (Robust Event-Guided Motion Deblurring), has successfully proven that disentangling modality-specific features can prevent this representation collapse during sensor corruption. However, RED is strictly limited to low-level vision tasks (outputting a single deblurred image) and relies on one-way robustness testing (corrupting only the Event stream).

This project significantly advances this frontier. We extend the hypothesis of disentanglement robustness into **High-Level Spatiotemporal Understanding**, specifically Visual Object Tracking (VOT), which requires continuous temporal ID persistence and complex geometric reasoning over long sequences. Furthermore, we move beyond RED’s single-modality masking by introducing a rigorous **bidirectional corruption matrix** and testing across three distinct architectural fusion paradigms.

**2\. Tech Stack and Methodology**

* **Target Benchmarks:** We will utilize two large-scale, tightly synchronized datasets specifically designed for high-speed tracking scenarios:  
  1. **VisEvent:** Featuring highly challenging scenarios like extreme low illumination and background clutter.  
  2. **EventVOT:** A dedicated large-scale benchmark for event-assisted visual object tracking.  
* **Architectural Paradigms:** To conclusively isolate the impact of our design, we will construct and evaluate three distinct fusion pathways:  
  1. *Early Fusion Baseline:* Stacking RGB and Event tensors at the input level.  
  2. *Late Fusion Baseline:* Fusing deep semantic features just prior to the tracking head.  
  3. *Disentangle \+ Fusion (Our Architecture):* A tripartite structure featuring one shared modality-invariant branch, an RGB-private branch, and an Event-private branch. The private branches extract orthogonal features before final fusion.  
* **Optimization Strategy:** The tripartite network will be trained end-to-end. The loss comprises the primary tracking loss alongside auxiliary disentanglement constraints:  
  $$Loss\_{total} \= Loss\_{task} \+ \\lambda\_1 Loss\_{contrastive} \+ \\lambda\_2 Loss\_{orthogonal}$$

**3\. The Two-Stage Execution Plan**

**Stage 1: Comprehensive End-to-End Training**

* **Objective:** Establish strong, uncorrupted performance baselines across all fusion paradigms.  
* **Method:** Train the Early Fusion, Late Fusion, and Disentangle \+ Fusion architectures from scratch on the VisEvent and EventVOT datasets.  
* **Success Criteria:** All three models must achieve state-of-the-art (or highly competitive) tracking accuracy under ideal, uncorrupted test conditions, proving that our auxiliary constraints do not hinder the network's baseline learning capacity.

**Stage 2: The Bidirectional Corruption Probe**

* **Objective:** Mathematically prove that only the disentangled architecture survives multi-modal degradation.  
* **Method:** Apply a systematic, bidirectional corruption matrix to the test sets for all three trained models:  
  * *RGB Degradation:* Inject extreme linear motion blur and dynamic range clipping.  
  * *Event Degradation:* Inject background flicker noise and sparse threshold shifts.  
* **Success Criteria:** 1\. Prove that Early and Late fusion models experience catastrophic failure when either modality is degraded, confirming latent entanglement.  
  2\. Prove that the Disentangle \+ Fusion model exhibits *isolated degradation*. Corrupting RGB should solely tank the activations of the RGB-private branch, leaving the shared and Event-private representations pristine to successfully maintain the bounding box track.

**4\. Expectations and Outcomes**

* **Primary Deliverable:** A highly robust Event-RGB tracking architecture capable of surviving localized, physics-based sensor failure in dynamic edge-case environments.  
* **Secondary Deliverable:** A comprehensive ablation study mathematically proving the fatal vulnerabilities of standard early and late fusion paradigms under bidirectional physical corruption.

**5\. Failure Modes and Strategic Pivots**

* **Failure Mode 1: Symmetrical Collapse.** The Disentangle \+ Fusion model fails just as severely as Late Fusion during the corruption probe, indicating the final tracking head is overfitting to the presence of both private branches.  
  * *Pivot:* Introduce an "Attention Dropout" mechanism during Stage 1 training, randomly zeroing out the private branches to force the shared branch to learn a self-sufficient baseline tracking representation.  
* **Failure Mode 2: Semantic Shredding.** Forcing strict mathematical orthogonality strips the private branches of so much contextual grounding that they become uninterpretable noise.  
  * *Pivot:* Abandon strict orthogonality for Predictive Coding. Alter the private branch loss function to predict the residual error of the shared branch, ensuring the private information tightly bounds the high-frequency details the shared branch missed.

**6\. Complementary Design: Real-World Bidirectional Corruption Matrix**

To ensure the robustness probe strictly mirrors physical hardware limitations, environmental extremes, and edge-case operational failures, we will programmatically inject the following physically grounded degradation models into the testing dataloaders. We explicitly avoid purely imagined or synthetic noise patterns in favor of known sensor vulnerabilities.

* **RGB Modality Corruptions (Simulating Optical & Sensor Limits):**  
  * **Aggressive Illumination Shifts (Low-Light & Overexposure):** Simulates the camera struggling with high dynamic range environments, such as a vehicle entering a dark tunnel or facing direct headlight glare. Implemented via non-linear gamma scaling and strict intensity clipping: $I\_{corrupted} \= \\text{clip}(\\alpha \\cdot I^\\gamma, 0, 255)$.  
  * **Linear Motion Blur:** Simulates the effect of high-speed ego-motion or fast target movement relative to the fixed exposure time of a standard global shutter. Implemented by convolving the image $I$ with a randomly parameterized linear directional motion kernel $K$: $I\_{corrupted} \= I \* K$.  
  * **Frame Dropping and Jitter:** Simulates common robotic pipeline bottlenecks (network latency or temporary sensor blackout) where the I/O fails to deliver a frame in real-time. Implemented by randomly dropping temporal frames entirely (replacing them with zero tensors) or temporarily duplicating the previous frame.  
* **Event Modality Corruptions (Simulating Neuromorphic Physics):**  
  * **Uniform Density Reduction (Low-Motion Sparsity):** Simulates scenarios where the tracked object briefly stops moving or perfectly matches the ego-motion of the camera, dropping the event generation rate to near-zero. Implemented by uniformly discarding a massive percentage of events across the temporal window with probability $p$.  
  * **Background Activity Noise (Low-Light False Positives):** In low-light environments, standard event cameras suffer from increased parasitic photocurrents, causing random pixels to trigger without actual spatial brightness changes. Implemented by injecting spurious events into the spatiotemporal volume following a Poisson distribution.  
  * **Hot and Dead Pixels (Hardware Degradation):** Simulates physical aging or manufacturing defects in the neuromorphic pixel array. A set of spatial coordinates $(x, y)$ is randomly selected to either permanently drop all incoming events (dead pixels) or fire continuously at maximum frequency regardless of motion (hot pixels).  
  * **Refractory Period Saturation (High-Speed Blanking):** When an edge moves excessively fast, the pixel circuit requires time to reset, causing it to blindly miss subsequent brightness changes. Implemented by discarding any events at coordinate $(x,y)$ that occur within a hardware-defined refractory temporal window $\\tau$ of the immediately preceding event.

**6\. Model Design**

All three models will use cross attention mechanisms to fuse the modalities to allow probing through attention scores.

* **Early Fusion Model**  
  For the early fusion model, we pass each modality into two very light conv layers respectively, then directly use cross attention for the subsequent layers, until the final detection head.  
* **Late Fusion Model**  
  For the late fusion model, we pass each modality into separate encoding branches, like ResNet-50, then at the very late stages, we fuse them using cross attention and send the fused feature to the detection head.  
* **Disentangle and Fusion Model**  
  This is the tripartite core. It explicitly isolates the physics of the sensors before mathematically fusing them.  
* Encoder Architecture: The network splits into three pathways.  
  1. Shared Semantic Branch ($S$): Processes both inputs to extract modality-invariant features (e.g., general object shape), enforced by contrastive loss.  
  2. RGB-Private Branch ($P\_R$): Extracts dense spatial texture, enforced by orthogonality to $S$.  
  3. Event-Private Branch ($P\_E$): Extracts temporal dynamics, enforced by orthogonality to $S$.  
* Fusing with attention mechanism: We use a Gated Injection Cross-Attention module. The stable, shared semantic representation ($S$) acts as the Query, while the two private branches act as the Keys/Values.  
  $$Attn\_R \= \\text{CrossAttention}(Q=S, K=P\_R, V=P\_R)$$  
  $$Attn\_E \= \\text{CrossAttention}(Q=S, K=P\_E, V=P\_E)$$

  The final output is a dynamically weighted sum:

  $$F\_{final} \= S \+ \\alpha \\cdot Attn\_R \+ \\beta \\cdot Attn\_E$$

**7\. Possible Orthogonality Design**

* **Strict Orthogonality Loss:** Directly force each private branch’s feature matrices to be far away from the shared branch.  
* **Soft Difference Loss:** Instead of strict orthogonality, you minimize the Frobenius norm of the dot product between the shared and private feature matrices: $\\mathcal{L}\_{diff} \= ||H\_{shared}^T H\_{private}||\_F^2$. It's easy to implement, but sometimes too weak to force true semantic separation.  
* **Adversarial Disentanglement:** You introduce a modality discriminator. The private branches are trained to help the discriminator guess the modality (RGB or Event), while the shared branch is trained to *fool* the discriminator. This is highly effective but introduces the classic GAN instability (mode collapse, balancing generator/discriminator learning rates).  
* **Mutual Information Minimization:** You use estimators (like CLUB or vCLUB) to explicitly minimize the mutual information between the shared and private spaces. This is theoretically the most rigorous method for "information disentanglement," but MI estimators are notoriously difficult to tune in high-dimensional deep feature spaces.


