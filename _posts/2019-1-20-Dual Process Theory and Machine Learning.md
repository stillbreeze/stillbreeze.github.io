---
layout: post
comments: true
title: Dual Process Theory and Machine Learning
excerpt: Can the influential ideas from Kahnemann and others' research about humans having two modes of thinking, one fast, one slow, be applied to current machine learning systems
published: true
---

The dual process theory, defined very generically states that our cognitive processes are composed of two types, Type 1 which is unconscious and rapid and Type 2 which is controlled and slow (also called System 1 & 2). The origins of these ideas go as far back as the 1970s and 1980s, but most non-experts like me know of these through the popular work and subsequent book (Thinking Fast and Slow) by Daniel Kahnemann. He often terms the two systems as intuition and reasoning. See [these examples](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow#Two_systems) for the two types of thoughts. To clarify, this theory isn't without faults and there's a lot of new work on more precisely defining the nature of these types or even proposing new theories which have more than 2 types [[^1]]. The link to a relatively recent paper about these discussions by Evans et al [[^2]] is given below. However, in this post, I would like to very briefly explore which properties of the theory have been or can be incorporated with machine learning. In contrast to previous posts, this will be a less rigorous and short article simply discussing some raw and speculative ideas. **It's possible that many questions I pose here have already been answered in cognitive science, neuroscience or psychology research, but I might be going off in the wrong direction**, in which case let me know in the comments. I also found a NIPS 2017 paper by Anthony et al from David Barber's lab at UCL called "Thinking Fast and Slow with Deep Learning and Tree Search" [[^3]] which explores the interplay between intuition and reasoning in RL systems through a new algorithm called Expert Iteration (EXIT). See [this blog](https://davidbarber.github.io/blog/2017/11/07/Learning-From-Scratch-by-Thinking-Fast-and-Slow-with-Deep-Learning-and-Tree-Search/) if you're interested in more a detailed explanation.

#### Characterizing Dual Process Theory (DPT)

In order to use DPT in ML, it should be helpful to enumerate some key properties of each type of thinking. The [Wikipedia page](https://en.wikipedia.org/wiki/Dual_process_theory#Systems) for DPT gives a list of items taken from Kahnemann's book. I have cherry-picked a few which I think have been/should be integrated with ML models.

|                   **Type 1**                  	|                    **Type 2**                   	|
|:-----------------------------------------:	|:-------------------------------------------:	|
|           Unconscious reasoning           	|             Conscious reasoning             	|
| Mostly linked to emotions ('gut feeling') 	|        Mostly detached from emotions        	|
|                 Low Effort                	|                 High Effort                 	|
|               Large Capacity              	|                Small Capacity               	|
|                   Rapid                   	|                     Slow                    	|
|              Default process              	|                  Inhibitory                 	|
|                 Nonverbal                 	| Linked to language or images in most people 	|
|               Contextualized              	|                   Abstract                  	|

#### Connecting DPT to machine learning

The difference in intuition vs conscious reasoning can be interpreted in ML in various ways. A lot of people like to think about intuition as priors (perhaps built into the structure of model) or as information we can access from a memory. But do such priors or embeddings retrieved from the memory also allow for other properties of Type 1 thinking mentioned above? And how do they compare with the Type 2 counterparts? Are there any other ways to incorporate the notion of intuition? Note that a completely separate and perhaps even more important question is of how to learn intuition from experience and data and what kind of models we need to enable this learning (evolution had millions of years to fine-tune our biology to make our brain's structure fit for learning from our world's data).

##### Voluntary vs involuntary thought

It's not clear to me, a single way in which two ML systems can differ in terms of being voluntary. Maybe involuntary and rapid action can simply be achieved by having a model which given a stimuli, accesses a memory indexed in a certain way (associative rules or context-specific rules). Maybe it need not be an explicit memory access, but an implicit distributed model which produces the rapid unconscious thought (the wiki article lists neural networks in Type 1 systems). In the last few years, the use of modular and even symbolic ML architectures for reasoning have become very popular. These models have decomposable modules which often deal with specialized inputs and grounded symbols and do explicit reasoning (like a 'locate' module only responsible for localizing a primitive in visual tasks). The difference between a distributed processing in a general neural network and such models where reasoning is explicit and information flow is constrained by the nature of the modules can perhaps be the difference between voluntary vs involuntary thought.

##### Role of intuitive psychology

I wrote earlier about the ideas of Lake et al (2016) which highlight how our psycho-social experiences shape our internal models. The 'gut feeling' we experience can be treated as our priors and incorporated in ML models as mentioned earlier  but it's tricky to define what kind of representations need to be used to enable such 'gut feeling' to be modelled.

##### Difference in effort and capacity

The two types of thinking also differ in the amount of computation and cognitive load. This can point to the fact that to replicate a Type 1 system in ML, it should be associated with lesser computation and lesser load on the working memory. The concept of effort can mean multiple things, but they can be more precisely defined than the previous properties. Similarly, capacity differences in the two types of thinking are also easier to define and implement. However an interesting question to ask here would be how exactly does a Type 1 system which involves low computation and is rapid, result in the general intelligence and high capacity we find to have as humans.


##### Type 1 as the default process

An important feature of the original DPT is that Type 1 thinking is the default due to its cognitive ease (also see [default mode networks](https://en.wikipedia.org/wiki/Default_mode_network)). For Type 2 thinking, we make a conscious effort to go beyond instinctive thoughts and reason about the task in a more involved way. But under which circumstances do we initiate the Type 2 thinking process instead of relying on our default? We know this will be a function of the task, the context in which we are performing it (eg our intrinsic or extrinsic motivations or the fact that we might have already tried our instinctive action which hasn't worked) and our past experiences relating to such tasks. I'm not sure how we can link this with the current ML models in an elegant way, but a naive solution can be to have an implicit or explicit meta-controller which is responsible for making this decision of switching from a default model to a dedicated model for reasoning. 

##### Type 2 is Verbal and visuospatial

This difference might be a consequence of the fact that most of the reasoning we do, is done through visual or verbal methods. But it also means that we can constrain the structure of reasoning-based models using our visual world or using the structure from language. Many reasoning based methods in ML are defined a particular way to exploit the regularities in the visual world or to constrain learning with the help of the syntanctic or semantic structures imposed by language. The fact that this is missing from Type 1 processes highlights the difficulty in defining concepts like intuition as they lead to representations which mostly aren't verbal or visual.

##### The mapping between tasks and type of thinking isn't fixed

Another important point I want to mention is that it isn't necessary that a slow Type 2 thought process requiring controlled reasoning and effort will always be so. We know from our experience that many Type 2 processes become Type 1 through experience and acclimatization. Placing your fingers a certain way when learning to play the guitar is obviously a Type 2 activity. But when learnt, we see experienced guitarists fluidly moving around their fingers and the task requires as less a cognitive load as a Type 1 task. RL systems do learn subsequent tasks at a faster rate, but I'm not sure if there's a switch occurring when they go from high effort and slow processing to that of low effort and rapid processing once they have learnt how to perform a task.

#### Conclusion

We haven't yet explored ML systems where these two types of systems co-exist and it would be very interesting to see what advantages this can offer. It might be the case that we don't actually have distinct systems within our brain to which intuition vs reasoning tasks can be localized separately. To point out one such possibility, many competing theories which extend DPT believe that the two types of thinking occur simultaneously [[^4]] \(which sounds super exciting to me in terms of the richness of learning which is possible). In any case, building such properties into ML systems from the top down will certainly open up new avenues and is worth discussing and experimenting with.


#### References

[^1]: [Diversity in reasoning and rationality: Metacognitive and developmental considerations](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/diversity-in-reasoning-and-rationality-metacognitive-and-developmental-considerations/AE82722C96F7E92E852030B7F09940F7)
[^2]: [Dual-Process Theories of Higher Cognition: Advancing the Debate](https://scottbarrykaufman.com/wp-content/uploads/2014/04/dual-process-theory-Evans_Stanovich_PoPS13.pdf)
[^3]: [Thinking Fast and Slow with Deep Learning and Tree Search](https://papers.nips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search.pdf)
[^4]: [Toward an integrative account of social cognition: marrying theory of mind and interactionism to study the interplay of Type 1 and Type 2 processes](https://www.frontiersin.org/articles/10.3389/fnhum.2012.00274/full)