---
layout: post
comments: true
title: Why Vision-Language Models Continue to Suffer from the Blind Parrot Syndrome
excerpt: How VLMs speak vision without seeing
published: true
---

<p align="center">
  <img src="/images/db_tweet.png" alt="Blind VLMs" width="60%">
</p>

Vision-language models have come a long way in the last 10 years. VLMs today are capable of explaining the image beyond captioning like its doing causal reasoning ("the man is running because the dog is chasing him"), generate fine-grained scenes specified in great detail, and even understand diagrams and do visual puzzle solving to some extent. For someone like me who spent years believing that these problems required _embodiment_ and _reasoning_ \- models that move, sense, and act in the world, towards goals that span different levels of abstraction, this progress looks miraculous. And yet, the miracle hides a trick. If you peel back the layers, much of this "understanding" doesn't come from better visual reasoning at all. It comes from advances in _language understanding_. Modern models have absorbed vast portions of our linguistic and cultural knowledge, giving them access to an encyclopedic sense of how the world usually works - without ever having to look, move, and operate in it. However, these models fundamentally still trip up in similar ways as their ancestor models did 5 years ago.

### **When Vision-Language Models Look Without Seeing**

VLMs still regress to what I call _blind parrots_ for unseen, complex problems, and repeat what they've read without doing any visual reasoning. Recent literature is filled with papers on VLMs behaving like bag of words and not respecting the object attribute relationship [[^1],[^2],[^3]], hallucinating objects not present in a scene [[^4],[^5]], being mediocre in spatial reasoning [[^6]], ignoring language cues like negation [[^7]] and over-relying on text modality, sometimes to the point that blind LLMs do better than a VLM on visual tasks [[^8],[^9]].

So why does this happen?

### **Vision is Hard and Language is a Crutch**

Let's go back to the tweet by Professor Dhruv Batra and explore those reasons for VLM failures on obvious tasks

- **Language imposes a strong prior** when used in VLM training. It is summarized knowledge, already symbolic in nature, trained with web-scale data of all human existence. CLIP and related architectures are left to learn largely statistical associations of vision mediated by language. This leaves little room during the learning process to create a sufficiently complex world model of the visual domain. Also, some of the hallucinatory properties and ignoring of words in VLMs are directly an inherited property of the language models.
- **Vision is hard**. Pixels have a low noise-to-signal ratio, require layers of transformations to reach the semantics, ViTs poorly condense fine-grained visual information, and most importantly, static 2D images are ultimately meaningless as the purpose of human vision is rooted in service of a much more general goal for an organism: survival. Moreover, unlike language where words can interact with words to generate complexity, vision must interact with action (like motion), thus requiring some kind of embodiment. And all this is missing in today's models. What results is shallow visual grounding between pixels and semantics, object hallucinations, and poor compositional and causal understanding.

It is not surprising that some of the core research happening in the field is to improve visual grounding of objects in the scene, make VLMs generalize to new compositions, explore neuro-symbolic reasoning, incorporate temporal and 3D vision into VLM models, and build environments where perception-action loops can be experimented. However, people who have been in the field for long will know, we've been down these roads before!

### **Should we build VLMs to explicitly do grounding and compositional reasoning?**

Before we conclude that VLMs must be explicitly taught grounding or compositionality, we should pause and look at what just happened in language and ask: What can we do differently this time?

Only a few years ago, we thought that logical reasoning, tool use, arithmetic, compositional generalization, even "theory of mind" would require explicit architectural changes or symbolic scaffolds. Then LLMs scaled, in data, parameters, and compute, and these behaviors **emerged**, not because we hardcoded them, but because the training dynamics, inductive biases, and data distribution _implicitly forced_ them to arise. I won't get into the philosophical question of whether the LLMs are _really_ reasoning. It's safe to say they're doing things which NLP researchers didn't think were on the horizon 10 years ago. So what about vision and language? A provocative question follows:  
**Why can't compositionality, grounding, and causal reasoning emerge in VLMs the same way?**

Maybe our visual systems in models haven't yet hit their _critical mass_. We're yet to get the scale or the pretext task from which all other competencies unfold.

Language had next-token prediction, a deceptively simple task that, when scaled, forced models to internalize syntax, semantics, pragmatics, and eventually something resembling reasoning. Maybe vision hasn't found its equivalent _foundation task_, that will pressure it to learn 3D structure, causality, and affordances implicitly.

We can speculate what that task might look like:

- **Temporal prediction in embodied worlds:** Predicting the next few frames in a physically consistent, dynamic environment might serve as the "next-token prediction" of vision.
- **Cross-view reconstruction:** Learning to reconstruct the world from multiple viewpoints or incomplete observations.
- **Causal visual dynamics:** Predicting how interactions between entities unfold under physical laws - "if I drop this, what happens?"
- **Video-to-language-to-action consistency:** Forcing models to maintain alignment across modalities through time, not just within a static frame.

It's plausible that the gap isn't conceptual but _scaling-based_. Our VLMs today are trained like CLIP - millions of independent (image, caption) pairs. But human perception develops through billions of temporally linked, physically constrained observations. Maybe the missing ingredient isn't new architecture, but the right kind of data continuity and predictive objective. Visual perception and action are co-constitutive. For today's VLMs, there's no analogous pressure: no reward for predicting affordances that matter for an agent in the world. But that can change.

### **Promising Solutions and Hybrid Paths**

While we wait for that moment, there are pragmatic avenues that help bridge both philosophies: explicit inductive bias _and_ emergent learning. After all, humans too rely on both. For example compositionality is often linked to the Binding Problem in cognitive science [[^10]] where even humans fail to compose correctly unless they are able to serially process visual information about individual objects. Similarly System 2 type reasoning in the Dual Process Theory also suggests the need to do symbolically grounded computation, which is how AlphaGeometry, from Deepmind got gold medalist performance in IMO [[^11]]. So here are some ad-hoc promising avenues for future work:

- **Object-centric / slot representations** as a substrate for compositional reasoning.
- **Grounded causal objectives** that reward models for predicting how changes ripple through a scene.
- **World models and embodied simulation**, allowing agents to "dream" forward in visual space.
- **Active perception** - models that choose where to look next, mimicking information-seeking.
- **Hybrid symbolic-relational layers** that encode structure but remain differentiable.
- **New diagnostic benchmarks** that force visual reasoning independent of linguistic priors.

### **Vision Hasn't had its true Foundation Model moment yet**

We keep training models to _associate_ pictures with text, when what we really need are models that _predict_ and _intervene_ - to see the world not as a set of pixels to describe, but as a process to anticipate and act in response.

When that pretext task arrives, that encodes physical continuity, temporal prediction, and goal-driven uncertainty, compositionality, grounding, and reasoning might not need to be explicitly modeled at all. They might simply _emerge_, as textual reasoning did in LLMs. Or very plausibly, we might discover that what we call reasoning in LLMs today is also a charade!

But for now, today's powerful VLMs can keep being eloquent about what they cannot see.

#### References

[^1]: [When And Why Vision Language Models Behave Like Bags Of Words And What To Do About It](https://arxiv.org/pdf/2210.01936)

[^2]: [CREPE: Can Vision-Language Foundation Models Reason Compositionally?](https://arxiv.org/pdf/2212.07796)

[^3]: [Mmcomposition: Revisiting The Compositionality Of Pre Trained Vision Language Models](https://arxiv.org/pdf/2410.09733)

[^4]: [Evaluating object hallucination in large vision-language models](https://arxiv.org/pdf/2305.10355)

[^5]: [A Survey on Hallucination in Large Vision-Language Models](https://arxiv.org/pdf/2402.00253)

[^6]: [SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](https://arxiv.org/pdf/2412.12693)

[^7]: [Vision-Language Models Do Not Understand Negation](https://arxiv.org/pdf/2501.09425)

[^8]: [Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](https://arxiv.org/pdf/2503.02199)

[^9]: [Vision language models are blind](https://arxiv.org/pdf/2407.06581)

<!-- [^10]: [Vision. A Computational Investigation into the Human Representation and Processing of Visual Information](https://mechanism.ucsd.edu/bill/teaching/f18/David_Marr_Vision_A_Computational_Investigation_into_the_Human_Representation_and_Processing_of_Visual_Information.chapter1.pdf)
 -->
[^10]: [Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem](https://arxiv.org/pdf/2411.00238)

[^11]: [Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2](https://arxiv.org/pdf/2502.03544)
