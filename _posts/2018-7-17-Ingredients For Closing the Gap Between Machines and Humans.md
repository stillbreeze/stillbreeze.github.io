---
layout: post
comments: true
title: Ingredients For Closing the Gap Between Machines and Humans
excerpt: Highlighting ideas at the intersection of cognitive science and machine learning by summarizing the work of Lake et al (2016) and its related paper commentaries
---

I have recently been going through some papers in the cognitive sciences, specifically related to cognitive theories and new ideas at the intersection of machine learning (ML) and human learning. The central question these papers aim to answer is a way forward to bridge the gap between current ML systems and the general learning abilities which humans possess. **The primary focus of this post will be to summarize one such paper along with some response commentaries it received from other researchers in the field**. Lake et al. in their paper, ["Building Machines That Learn and Think Like People"](https://arxiv.org/pdf/1604.00289.pdf), identify certain ingredients of human cognition which can help ML researchers realise systems which learn like humans. The remaining post summarises some interesting ideas from the paper and the commentary, but do check out [[^1]] for the full paper and [[^2]] for all the related commentaries.

#### Core ingredients of human intelligence
An overview of the described human-like learning characteristics from Lake et al:

1. Developmental start-up software
	1. Intuitive physics
	2. Intuitive psychology
2. Learning as rapid model building
	1. Compositionality
	2. Causality
	3. Learning-to-learn
3. Thinking Fast
	1. Approximate inference in structured models
	2. Model-based and model-free reinforcement learning

##### 1. Developmental start-up software

The authors contend that humans have a foundational basis of understanding certain concepts like set operations, mechanics, geometry and agency. For example, a small infant is capable of identifying distinct objects, anticipating how rigid objects move under gravity, how solids differ from liquids when touched or how human agents differ from inanimate objects. All these examples are clubbed up under the foundational start-up software which comes intuitively to a child and on top of which further experiences build more knowledge. 

###### 1.1. Intuitive physics

Children have a developed intuitive physical state representation which gives an approximate, probabilisitc and oversimplified account of the the physical world and its interactions. Many recent works have tried to embed deep learning models within a physics simulation engine from which it can learn this notion of intuitive physics, but how well it incorporates the physical rules of world, whether it can learn with as few experiences as humans do and how we evaluate what they have learnt are all challenging problems.

###### 1.2. Intuitive psychology

How children perceive world agents like other humans or animate objects and how they react to these agents gives us a view of how psycho-social experiences shape our intrinsic mind models. Lake et al. give examples of how children associate negatively with an agent who blocks a positive action based on cues. But the no of cues need to scale rapidly as situations become more complex for this to be plausible. Alternatively, such reasoning can be thought of as a generative model of actions where the child is seen as optimising for some goal through mental planning (like that of an MDP or POMDP). However research connecting these psycho-computational theories to deep learning models have just begun (see [[^3]] and [[^4]]).

##### 2. Learning as rapid model building

Humans have an amazing capacity for generalizing with few examples. We can see, relate, imagine and describe new concepts and make plausible inferences about them. Moreover, there is considerable evidence that this few-shot learning occurs on top of domain knowledge of various other classes of concepts (we can mentally picture a monkey with wings roller skating on the road because we have previous knowledge about the mentioned objects and their functional nature). The question is how to integrate various domain knowledge into current ML models to enable rapid model bulding.

###### 2.1. Compositionality

Compositionality is the mechanism which allows humans to build complex representations by composing multiple primitives. Therefore, instead of individually learning complex concepts which is combinatorially expensive, they are learnt as general composition of simple representations. This allows faster few-shot learning of novel concepts. Many recent papers explore the cognitive theories ([[^5]] and [[^6]]) and computational models ([[^7]], [[^8]] and [[^9]]) in the machine learning for concept composition, but of course a lot of remains to be understood. 

###### 2.2. Causality

Causality refers to the generative process by which a certain prediction or observation is produced in humans. Novel few-shot learning is often dependent on the nature of causal models we have learnt in the past. Causality acts as a glue for binding together various concepts and events in order to constrain our learning towards the real-world observations. It is important to note however, that not all generative models in machine learning are necessarily causal as they might not have anything to do with the actual process of generating that data.

###### 2.3 Learning-to-learn

Many priors and inductive biases humans gain during the learning of one task is often useful for newer tasks. Learning-to-learn is thus the ability to transfer representation and computational structure to solve novel tasks. This problem has been discussed for long in the machine learning community (Jurgen Schmidhuber did a lot of early work in the 1980s) and a lot of new ideas in deep RL and supervised transfer learning have pushed the performance benchmarks further (also see [this](https://github.com/floodsung/Meta-Learning-Papers) list of papers on meta learning). Nevertheless, modern ML systems don't learn as rapidly and flexibly as humans do and meta-learning will certainly have an important role to play here.

##### 3. Thinking Fast

Given that humans seem to have complex and structured models which allow for rapid generalization (previous three characteristics), it's even more remarkable that the inference in these models is extremely fast. Deep learning based approaches are often advantageous due to their fast inference times and scalability and can form a viable basis for more human-like ML systems.

###### 3.1. Approximate inference in structured models

It is vital for any human-like ML model to perform approximate inference as calculating the probability distribution over the entire search space is almost always intractable. Some cognitive theories posit that humans perform approximate bayesian inference using stochastic sampling methods like Monte Carlo sampling. Inductive biases are also evoked for facilitating rapid hypothesis selection in addition to hypothesis evaluation. For example, we know the answer to the question "how old is that tree?" is a number even though we may not know the correct answer or a never-before-seen object with wheels can be moved around even though we haven't interacted with it yet ()it might not even move in reality but we still make these inferences). In the recent ML literature, many methods learn to do amortised inference in graphical models and the work done in probabilistic inference in generative models or differential programming are exciting avenues for the integration of deep learning and structured probabilistic models. 

###### 3.2. Model-based and model-free reinforcement learning

There's significant evidence that the human brain uses fast model-free algorithms like the ones used in DQN models. However many cognitive capabilities which we exhibit also point to the presence of model-based learning. For example, for a given state-action environment, our brain can flexibly adapt to optimise for different reward signals without re-learning. This highlights our capacity to build a cognitive map of the environment and re-use it for different end goals. Thus, it is necessary for ML systems to allow for both model-free and model-based mechanisms. 

#### Response commentary from Behavioral and Brain Sciences Journal

There are 27 commentaries on the above summarized Lake et al. paper, but I have chosen a select few from them based on what I found interesting and have tried to condense them into few bullet points. The purpose here is to stimulate thoughts in different broad directions.

##### 1. The architecture challenge: Future artificial-intelligence systems will require sophisticated architectures, and knowledge of the brain might guide their construction by Baldassarre et al

- Developing new architectures is essential for human-level AI systems.
- Looking at the brain can provide guidance as to which architecture spaces to look at for navigating through the tons of possible architectures. Eg. Cortex is organised along multiple cortical pathways which are hierarchical with higher ones focussing on motivation information and lower ones on sensation.

##### 2. Building machines that learn and think for themselves by Botvinick et al

- Agree with the list of ingredients, but focus should be on autonomy to reach these goals (agents learn their own internal models and how to use them instead of relying on human engineering).
- Learning agents should be able to capable across multiple domains without needing too much of priori knowledge.
- The idea is to use high-level prior knowledge like general structures about compositionality or causality (just like translational invariance was built into CNNs) along with large-scale and general architectures and algorithms like attentional filtering, learning through intrinsic motivation, episodic learning and memory augmented systems.
- Models should be calibrated not just to individual tasks but to a distrbution of tasks, learnt through experience and evolution. Thus, autonomously learning of internal models such that these models can be shaped by specific set of tasks is advantageous.
- Autonomy also depends on control functions (processes that use the model to make decisions). Even these control functions should co-evolve with models over time, hence agent-based approaches are important to develop.
- Model free methods might be primarily important. It's premature to relate them to a supporting role.

##### 3. The humanness of artificial non-normative personalities by Kevin B. Clark

- Cognitive emotional behaviour and non-normative (unique) personalities and in turn, dynamic expression of human intelligence and identities is a key aspect of being human which is overlooked in Lake et al.
- Attributes like resoluteness, meticulousness, fallibility, natural dispositions etc are all very human traits and must be accounted for in an artificially intelligent agent in order to realise their effects on learning and in order to prevent unwanted machine behaviour.

##### 4. Evidence from machines that learn and think like people by Forbus and Gentner

- Analogical comparison are an important part of human reasoning and might be better than learning structured relational representation.
- Qualitative representations, not quantitative simulations are the main ingredients of conceptual structure in the brain. Actual dynamics of the physics might not be known or even be encoded in the model, but just a qualitative experience is needed. Hence the Monte Carlo simulation of the kind used in lake et al. (another 2015 paper) might not work.

##### 5. The importance of motivation and emotion for explaining human cognition by Güss and Dörner

- Lake et al focus only on cognitive factors and misses out motivation and emotion. Motivation and diverse exploration (seeking uncertainty in order to minimise it later on) and emotion lead to many human behaviours which interacts with the cognitive processes.

##### 6. Building on prior knowledge without building it in by Hansen et al

- Compositional approach is limited because it downplays the complex interaction of multiple contextual variables related to the various tasks where the representations are used. Not committing to compositionality provides more flexible ways of dealing with learning complex representations.

- An important direction to explore is how humans learn from a rich ensemble of multiple varying, but partially related tasks.

- Meta learning of these related sub-tasks can be done, with the meta-tasks becoming more general (eg give an explanation for your behaviour, incorporate comments from a teacher etc.) which will not rely on a startup software which requires domain-specific prior knowledge.

##### 7. Benefits of embodiment by MacLennan

- Lake et al focus on the startup software but neglect the nature of the software or how it is acquired. For understanding intuitive physics and physical causality, embodied interaction of a organism with an environment serves as a guide for higher order imagination and conceptual physical understanding. Simulations, in principle, can help in developing similar competencies, but generating simulations with enough complexity is difficult. 

- Explicit models are the ones which scientists construct in terms of symbolic variables and reason about them discursively (including mathematically). Implicit models are constructed in terms of large no of sub-symbolic variables which are densely interrelated (like a neural network). Implicit models allow for emergent behaviour and are more likely to be relevant to the the goal of human-like learning.

##### 8. Autonomous development and learning in artificial intelligence and robotics: Scaling up deep learning to human-like learning by Oudeyer

- Curiosity, intrinsic motivation, social learning and natural interaction with peers and embodiment are interesting areas to probe.

- Many of the current systems have manually specified, task-specific objective. Many are learnt offline, that too on large datasets. Whereas, human learning has open ended goals and explores various skills. It is online and incremental in nature and involves free play.

- Human learning happens in the physical world under constraints of energy, time and computation. Thus, embodiment is crucial for learning. Sensorimotor constraints in the models can simplify learning.

##### 9. Crossmodal lifelong learning in hybrid neural embodied architectures by Wermter et al

- Lifelong learning through experiencing the 'world' is the next big direction for ML systems. Additionally, these models should facilitate cross-modal learning to make sonse of the multimodal stimuli of the environment.

- Startup-software is tightly coupled with the general learning mechanisms of the brain. Past research suggests that architectural mechanisms like different timings in information processing in the cortex, foster compositionality, that in turn enables more complex actions.

- Transfer learning shouldn’t merely be switching between modalities, but integrating multiple modalities which are richer than the sum of their parts.

#### Acknowledgement

A big thanks to [Andrew Lampinen](http://web.stanford.edu/~lampinen/) for helping me access the paper commentaries. 


#### References

[^1]: [Building Machines That Learn and Think Like People](https://arxiv.org/pdf/1604.00289.pdf)
[^2]: [Related Commentaries: Building Machines That Learn and Think Like People](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/building-machines-that-learn-and-think-like-people/A9535B1D745A0377E16C590E14B94993#fndtn-related-commentaries)
[^3]: [Intrinsic motivation, curiosity and learning: theory and applications in educational technologies](https://hal.inria.fr/hal-01404278/document)
[^4]: [Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents](https://arxiv.org/pdf/1712.06560.pdf)
[^5]: [Cognitively Plausible Theories of Concept Composition ](https://pdfs.semanticscholar.org/6bd9/fa9aad10e0edd965c9bb43882a487c875d08.pdf)
[^6]: [Conceptual Versus Referential Affordance in Concept Composition](https://link.springer.com/chapter/10.1007/978-3-319-45977-6_10)
[^7]: [From Red Wine to Red Tomato: Composition with Context](http://openaccess.thecvf.com/content_cvpr_2017/papers/Misra_From_Red_Wine_CVPR_2017_paper.pdf)
[^8]: [Generative Models Of Visually Grounded Imagination](https://arxiv.org/pdf/1705.10762.pdf)
[^9]: [Attributes as Operators](https://arxiv.org/pdf/1803.09851.pdf)