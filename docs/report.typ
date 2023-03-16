#import "report_template.typ": *

#show: project.with(
  title: "MotorNet-Probabilistic: Stochastic Model-based RL for Reach Control of a Realistic Arm",
  authors: (
    (name: "Amirhossein Asadian", email: "aasadia3@uwo.ca", affl: "Western Institute for Neuroscience"),
    (name: "Pranshu Malik", email: "pranshu.malik@uwo.ca", affl: "Western Institute for Neuroscience"),
  ),
  abstract: [
    We are inherently able to produce movements that help us achieve our goals in the world. In order to do so,
    we make use of our knowledge of the environment and our own body to produce optimal-looking movements. This self- and
    environmental-knowledge can be thought to be encoded in the form of internal models, which are central to 
    inferring sensorimotor control. These models can be forumlated various ways, implicitly or explicitly, depending on 
    the modelling perspective and level of abstraction (e.g. control-theoretic modules, algorithmic procedures, or 
    implicit transformations in artificial neural networks). This work aims to externalize the iterative internal models 
    for the control of a realistic arm model by using the model-based reinformcent learning (RL) framework, 
    which is contrary to the predominant and performant recurrent neural network (RNN) controllers wherein the 
    internal models and control hierarchy are entirely obscured. In this study, we present the integration of a 
    realistic and differentiable arm model and then train a baseline model-free stochastic RL policy to control the arm. 
    We also propose a model-based RL formulation, through which we may be able to ask further scientific questions on 
    the organization of naturalistic control, as well as achieve better artificial control methods. Altogether, these 
    form the first ideas for MotorNet-Probabilistic (or #smallcaps("MotorNet-Pro") in short).
  ],
  keywords: [sensorimotor control, model-based continuous feedback control, stochastic policy]
)

= Introduction
o	Why is it an important problem?
o	Why can't any of the existing techniques effectively tackle this problem?
o	What is the intuition behind the technique that you developed?

Motor behaviors that may come intuitively to us are not easy to achieve synthetically. The body is a complex system, all
the way from the anatomy -- featuring underactuated and nonlinear musculoskeletal dynamics -- to noisy and delayed 
sensorimotor modalities -- with indirect and distinct feedback spaces as well as distributed control structures.
To overcome such complexities, our bodies use internal models to predict and control movements in an adaptive 
manner, producing optimal and robust control even in unpredictable and uncertain environments @InternalModels. However, 
our model-based control policies, can also display model-free _strategic_ changes to adapt to 
unexpected environmental conditions even though reaction to perturbations through reflexes is model- and policy-based 
@CrevecoeurRobustControl. This shows the need for a more general framework and formulation that models the organization 
of control as not as a single model-based policy output but also policy modulation, adaptation (e.g. continual 
learning), and online multi-level planning and control. Added to that, the framework should be grounded in stochastic 
control, as this is a key feature of naturalistic movements.

To this end, most neuroscientifically-motivated modelling attempts have included an RNN-based controller, which lends 
itself to a neuron-population level analyses of activity patterns which are similar to what is done on experimental
recordings from the motor-related cortical areas. MotorNet by @MotorNet #label("motornet") is one such realistic arm 
model coupledwith a controller that is trained in delayed continuous feedback and continuous action spaces, following 
the optimal feedback control (OFC) framework laid out by #cite("OFCTodorov", "OFCScott"). This is a promising direction, 
especially when studying neural dynamics in cortical control, however, we are interested in the organizational aspect of 
motor control in a more _interpretable_ fashion, for which the model-based RL framework is more suitable. In a study by 
@RNN-SAC, performant control is elegantly achieved with an RNN actor in soft actor-critic (SAC) by @SAC so that the 
policy can learn to do online control by maintaining a stateful representation of the ongoing movement and task demands. 
However, similar to MotorNet, the purpose of this controller was to investigate proximity of neural activity in the \
RNN-pool and with neural recordings of the motor cortex. A study by @BiomechRL also trained a SAC policy, but for a much 
simpler arm model, and was focussed on explaining certain movement laws that arise through in the stochastic policy that 
prove certain properties of the assumed noise model, proposing that the biomechanical system can be understood at a 
simpler scale. However, we are interested in how control is produced in the first place, which would also help address many open questions on the same front as well as provide clues on how to achieve better robot control.

= Problem Formulation
Hi
o	Brief review of previous work concerning this problem (i.e., the 4-8 papers that you read)
o	Brief description of the techniques chosen and why
o	Describe the technique that you developed
o	Brief description of the existing techniques that you will compare to

Both forward and inverse models are used in control: policy usually acts as an iterative inverse model that is directed 
towards the goal. We would like to fuse the feedforward capability (inv model) to use feedback control (forward model) 
to be able to create an online optimal robust controller; for this, the policy should have distilled the knowledge about 
the stateprediction and subsequent control -- the dreaming architecture is good for that; 

For that, it should have an idea of what future states (short term) I might be in, and thus plan for those ... miniplans/
chunks. For this, a multilevel policy net can be trained; one for the objective space and short-term plan and another 
for the control in that; this enables implicit chunking and subsequent local robust control; this is also a good way to 
train an RNN to replicate this -- easier way to bake-in interpretability into the RNN models.

Planning and control in latent feedback space: all pixel and dreaming models will be useful. Planning predicts the 
future statespace of potential interest and then control is done on that space by understanding the reward/objective 
landscape for the short landscape; this iterates cleanly as states progress, quite naturally.

Latent predicted future feedback manifold/landscape on which the actor level of the policy is trained to control; 
@RNNDynamics (among various others) discuss this in RNNs and this is also a dominant idea in neuroscience literature 
under the banner of optimal feedback control #cite("OFCTodorov", "OFCScott").

#grid(
  columns: (2),
  rows: (auto),
  column-gutter: 0em,
  row-gutter: 2em,
  align(center+horizon)[
    #hide[space] // 1em
    #figure(image("./assets/plant.svg", width: 80%),
    caption: [#link("motornet")[MotorNet] Plant])
  ],
  grid(
    columns: 1,
    rows: (3cm, 3cm),
    column-gutter: 0em,
    row-gutter: 2em,
    figure(image("./assets/momentarms.svg", width: 100%), caption: [Moment Arms]),
    figure(image("./assets/musclelengths.svg", width: 100%), caption: [Muscle Lengths])
  )
)
#v(1em)

Noteably, the forward state prediction can act at different timescales, a.nd for timepoints with complex dynamics, the 
states can evolve with a zoomed in parametrization so that objectively lesser feedback space is included in the 
plannning/inferring control, but it leads to more precise manuevers. This is similar to how numerical integration of 
ODEs work; closer to more error-prone timepoints, there are more iterations of the solver. It seems logical that online 
optimal feedback control should do the same.

Should also relate to some #emph[knowledge distillation] ideas where the forward/transition model applies to first level 
of the policy and the reward model applies to the second level of the policy to learn an evolving local reward 
landscape. Should perturbations happen, and the actor be thrown outside the current state landscape, it will 
automatically have to reiterate the bilevel processing, but if the perturbation is small enough and it fits the current 
local computation, then it should be able to produce a robust correction based on the reward landscape with subsequent 
recomputation at the first level within a few more steps. Stiffness control can also be introduced during training to 
include the ability of the model to adapt to new environment dynamics #cite("Shadmehr1994", "Conditt1997") or also 
produce like in @CrevecoeurRobustControl. Since the literature on this front is sparse in neuroscience, modeling this 
adhoc is one way how such a model can be used immediately in sensorimotor neuroscience to generate new hypotheses and 
experiments, test and then update our understanding. Similar to the model-free response above, we can similarly also use 
this bilevel policy to study the emergence of multiple strategies (and their selection) in motor control.

#align(center)[
#figure(image("./assets/motornet_pro_architecture.png", width: 50%), 
caption: [#smallcaps[MotorNet-Pro] Architecture]) <motornet_pro_architecture>
]

- All of these "snapshots" can be replayed to train an RNN, which will regularize learning of representations and provide us some control over interpretability of the same -- going from encoded and learnt rules to a universal differential equation.
- Such a control policy can then also be applied to robots and control of other limbs and the whole body.

// todo: why not train 2level policy directly on the latent space, and then how much different would this be from ALM?
// todo: need to see if we need a time-varying time-stepped feedback latent space model; how would we do that??
// should provide intermediate and pre-timept true feedback to train as we are making errors... (non-biological but)
// maybe only briefly mention if we even need this, since over time and experience you would tend to sample inter-
// mediate feedback, only that it won't be timestamped...

== Related Work
This has been done. More conventional and early approaches first began with the @DDPG algorithm, then
many came on to use MPC to have an implicit policy requiring a sub-routine to get the current
action by solving the optimal-control problem (as a nonlinear program NLP) at each time step (or state). 
Even though MPC-based policy and value functions approximation may offer a high explainability
about the policy behaviour in addition to being equipped with a broad set of theoretical tools for formal 
verification of safety and stability e.g. @PETS, it is not a good fit for our problem.
This is computationally expensive and especially not meeting our criteria for explicit representation of the 
policy and internal models.
#text(fill: red)[Give MPC formulation and parametrization by policy param $bold(theta)$]. 
By using off-the-shelf non-differentiable solvers, we have the added disadvantage of losing the ability to 
backpropagate gradients through the policy. #text(red)[give citations too: PSRL etc.].

Some model based RL approaches: #cite("MBPO", "MAAC", "PSRL", "SVG"). Some other ideas are: 
#cite("ModelBasedImplicitDifferentiation"). @REDQ also looks promising, but Pytorch implementation -- needs
to be done in Tensorflow. After this, this would be our goto baseline for not only comparison but also 
gaining insights into the learnt policy. What is suggested by and contemporary motor control theories @CompetingPlans 
is that there is benefit in including a small ensemble of planning models to eventually reduce the variance. This 
could also allow for modelling several strategies in these policy/actor models.

There are also image-based control algorithms that infer from a latent state, this should also generally apply,
however, to our best knowledge the literature does not seem to have shown that, particularly it would be nice to
see if the latent space is better to generalize #cite("SLAC", "PlaNet", "DRQv2"); pixel-to-control. However, most 
of engineering innovations made in the algorithms (e.g. displacing image by $plus.minus 4$ pixels) seem to also not 
fit any biological detail involving change of feedback structure or processing. But, noteably, PlaNet @PlaNet makes 
an important Recurrent State Space Model (RSSM) that predicts forward in latent space split the state into stochastic 
and deterministic parts, allowing the model to robustly learn to predict multiple futures. A different, but similar 
model strategy, Dreamer, was developed by @Dreamer that, through its formulation, is able to analytically calculate
and exploit gradients from state transitions to speed up the RSSM-model learning. This was implemented for real-time, 
real-world learning in robotics by @DayDreamer. Adapting these latent-space ideas for the bilevel distilled policy 
network is a promising direction.

Along the topic of planning and control in latent space, work by @ALM on aligning these latent space models each of
which were previously with their own auxiliary objectives, making the submodel alignment unclear. Single objective 
which jointly optimizes a latent-space model and policy to achieve high returns while remaining self-consistent.
This is a very close analogue to our bilevel policy formulation, as it a model that predicts in representation space
for the feedback instead of high-dimensional observation space, and a policy that acts based on those representations.

This is different than the proposed conceptual formulation @conceptual_policy is the implicit inclusion of online 
planning through continual state prediction in the latent space, to a variable extent conditioning the actor on the 
goal(s) thereby setting the context for action. This naturally falls into the scope of, however, more thought is 
needed on the mechanism and appropriate setup and formulation for adaptive contraction and expansion of feedback 
prediction space. An alternative would be to possible that the transformer and variational autoencoder (VAE) based 
models.

MBPO: ensemble of models trained using MLE, trained on environment data only and then used to produce cheaper
rollouts in each episode to train the policy further from the sampled states. (some criticism from MAAC and PSRL).
When the model class is misspecified or has a limited representational capacity, model parameters with high 
likelihood might not necessarily result in high performance of the agent on a downstream control task 
@ModelBasedImplicitDifferentiation. Nishkin et al. propose a method to address this issue by ... What it may
mean for us...

== Exploratory Formulation
Based on the recent innovation novel algorithm Action Chunking with Transformers (ACT) which reduces the effective 
horizon by simply predicting actions in chunks, we would also like that as a generalization of the policy formulation
@ACT. This can also be supported by the neuroscience literature in the sense that a near short horizon plan is known
at the time of execution although the cost landscape for actions beyond the current may evolving in an online manner.
Bringing this formulation into the RL framework can be hard and for now is beyond the scope of this report. But this
is something we will definitely try to incorporate simply in the future.

Similar to how optimal control is done in the brain, with a higher-level context coming in, also recently demonstrated
to be critical to a @RNNPrep. This highlights the need for a plan and also the fits the introspection of having 
constructed a general plan or strategy of movement before.

In our model, actions are defined as the activation of each muscle, which results in a 10-dimensional action space. The 
state is determined by a variety of observations, including proprioceptive feedback (muscle length and velocity for each 
muscle), visual feedback (position of the arm), perceived target, and muscle activations (affordance copy of each 
muscle), resulting in a 34-dimensional state space. To account for the delay in information processing in the central 
nervous system, each feedback changes the state after a certain amount of time. Specifically, we have set a 
proprioceptive delay of 20 ms and a visual delay of 70 ms. Additionally, to account for noise in neural pathways, we 
have added a normal noise with a standard deviation of 0.01 to each feedback information (proprioceptive feedback, 
visual feedback of arm location, and visual feedback of target location).

#align(center)[
#figure(image("./assets/latent_ofc_policy.png", width: 60%), 
caption: [Latent OFC Policy]) <conceptual_policy>
]

= Baseline Algorithm and Considerations
As a starting point, we have decided to implement the soft actor-critic (SAC) algorithm for policy optimization. SAC is 
a popular model-free deep reinforcement learning algorithm used for continuous control tasks. It is based on the maximum 
entropy reinforcement learning framework, which aims to maximize the expected return of a policy while also maximizing 
the entropy of the policy distribution. By maximizing the entropy, SAC encourages exploration and prevents the policy 
from getting stuck in local optima.

At each iteration, SAC performs two main steps: policy evaluation and policy improvement. In the policy evaluation step, 
SAC estimates the state-action value function $Q_pi$, which represents the expected return starting from a given 
state-action pair under the current policy. The $Q$-function is updated using the Bellman backup operator:

$ Q(s,a) = EE [r + gamma V(s')] $

where r is the immediate reward, $gamma$ is the discount factor, and $V(s')$ is the value function of the next state. 
The policy improvement step involves training a stochastic policy that minimizes the expected Kullback-Leibler (KL) 
divergence between the current policy and the exponential of the Q-function minus a value function $V_pi$. The objective 
function for policy improvement is given by:

$ J(π) = EE [Q_pi(s,a) - alpha log(pi(a|s))] $

where $alpha$ is a temperature parameter that controls the balance between maximizing the expected return and maximizing 
the entropy  #cite("SAC", "MBPO").

We have chosen SAC because it is easy to implement and has shown promising results in a variety of domains. While we 
will be using SAC as our baseline algorithm, we are also considering the approach proposed by Chen et al. (2021) for 
future use. They have suggested an ensemble of SAC models combined with in-target minimization as an improved method, 
and we will keep this in mind for potential future optimization.

It is worth noting that our goal is not only to achieve high performance levels and quick convergence but also to 
develop a model-based policy that is interpretable and can provide insights into the underlying processes. We believe 
that this combination of performance and interpretability could serve as a valuable oracle for recurrent neural networks 
and model trans-cortical computation for movement control, including long-latency reflexes (LLRs) and optimal and robust 
voluntary motion.
// We are doing this first.
// We adopt soft-actor critic (SAC) as our policy optimization algorithm. SAC alternates between a policy evaluation 
// step, which estimates Q$pi = E[]$ using the Bellman backup operator, and a policy improvement step, which trains 
// an actor π by minimizing the expected KL-divergence Est∼D[DKL(π|| exp{Qπ − Vπ})] #cite("SAC", "MBPO").
// SAC going forward is the easiest to implement; @REDQ an ensemble and in-target minimization outperforms SAC using 
// simple methods with theoretical guarantees, making it the (current) ideal model-free candidate for our problem.
// Here, note that what we are looking for is not only performance levels and how quickly they are attained, but also
// the model-based policy's ability and interpretability. Later this combination could be used as an oracle for RNNs
// and model trans-cortical computation for movement control (LLRs, optimal and robust voluntary motion, etc.).
// We are doing this first.
// We adopt soft-actor critic (SAC) as our policy optimization algorithm. SAC alternates between a policy evaluation 
// step, which estimates Q$pi = E[]$ using the Bellman backup operator, and a policy improvement step, which trains 
// an actor π by minimizing the expected KL-divergence Est∼D[DKL(π|| exp{Qπ − Vπ})] #cite("SAC", "MBPO").
// SAC going forward is the easiest to implement; @REDQ an ensemble and in-target minimization outperforms SAC using 
// simple methods with theoretical guarantees, making it the (current) ideal model-free candidate for our problem.
// Here, note that what we are looking for is not only performance levels and how quickly they are attained, but also
// the model-based policy's ability and interpretability. Later this combination could be used as an oracle for RNNs
// and model trans-cortical computation for movement control (LLRs, optimal and robust voluntary motion, etc.).

// todo: create template function for the same (update algo and add line counters; manage indentation)
#pad(left: 30%)[
  #line(length: 8cm)
  #v(-1em)
  #strong[Algorithm 1:] Soft Actor-Critic
  #v(-0.8em)
  #line(length: 8cm)
  #v(-0.8em)
  Initialize parameter vectors $psi$, $macron(psi)$, $theta$, $phi.alt$.\
  #strong[for] each iteration #strong[do]\
  #hide("    ")#strong[for] each environment step #strong[do]\
  #hide("    ")#hide("    ") $bold(a)_t tilde.op pi_phi.alt(bold(a)_t|bold(s)_t)$\
  #hide("    ")#hide("    ") $bold(s)_(t+1) tilde.op p(bold(s)_(t+1)|bold(s)_t, bold(a)_t)$\
  #hide("    ")#hide("    ") $cal(D) arrow.l cal(D) union {(bold(s)_t, bold(a)_t, r(bold(s)_t, bold(a)_t),
  bold(s)_(t+1))} $\
  #hide("    ")#strong[end for]\
  #hide("    ")#strong[for] each gradient step #strong[do]\
  #hide("    ")#hide("    ") $psi arrow.l psi - lambda_V hat(nabla)_psi J_V(psi)$\
  #hide("    ")#hide("    ") $theta_i arrow.l theta_i - lambda_Q hat(nabla)_theta_i J_Q(theta_i)$ for $i in {1, 2}$\
  #hide("    ")#hide("    ") $phi.alt arrow.l phi.alt - lambda_pi hat(nabla)_phi.alt J_pi(phi.alt)$\
  #hide("    ")#hide("    ") $macron(psi) arrow.l tau psi + (1 - tau)macron(psi) $\
  #hide("    ")#strong[end for]\
  #strong[end for]\
  #v(-0.8em)
  #line(length: 8cm)
  // optional: caption
]

== Preliminary Results
We extensively trained our model over 10,000 episodes to achieve the target within one second. Figure 5 showcases the 
learning curve, and the best episode rewarded our model with -249.65. Interestingly, we observed optimal performance 
after 2000 episodes, but then a gradual decline occurred. This decline may be attributed to "catastrophic forgetting," 
where the model's success causes it to forget what failure looks like. As a result, the model predicts high values for 
all states and features, regardless of their relevance.

When the model encounters unexpected situations with incorrect predicted values, the error rate can be high, and 
recovery can be challenging. Additionally, the model may incorrectly link features of the state representation, making 
it difficult to distinguish between various parts of the feature space. This creates unusual effects on the model's 
learning about the values of all states. While the RL model may behave incorrectly for a few episodes before relearning 
optimal behavior, it may also break down entirely and never recover.

Catastrophic forgetting is an active research area, and one potential solution is to set aside some percentage of replay 
memory with the initial poor-performing random exploration. We are actively considering various approaches to tackle 
this issue.

Other implementations exist for reaching in environments with more than one arm, such as the one using 22 double-joint 
arms with a 33-dimensional observation space (cite git). However, they control simplistic arms lacking realistic and 
non-linear features. They utilize a single agent to control multiple arms in order to promote generalization. In 
contrast, we are exploring the use of the DDPG algorithm to enable the model to reach different targets from varying 
starting positions. Our primary objective is to gain insight into the internal models of this agent as the model learns 
to reach different locations in a 2D space.

#align(center)[
#figure(image("./assets/sac_learning_curve.svg", width: 50%), 
caption: [SAC Learning Curve]) <sac_learning_curve>
]

= Discussion and Future Work
Many possibilities. First to improve the baseline, we can do Dyna-style @SuttonDyna-Q virtual rollouts
to improve the policy, however, doing this early on may be detrimental to the learning as the
model will not be accurate enough. Also adding virtual rollouts to the replay buffer may be 
problematic for the same reason. Some theoretical work on need- and risk-aware model learning also exists, 
such as @PAML, but it will need to be adapted to evolving stochastic continuous control framework.
Overall, this approach is mostly an empirical tweak and its efficacy will depend on the extent of fine-tuning
the baseline algorithm and its features. One missing thing is the inclusion of signal dependent noise in the 
MotorNet plant model.

We can use the same points of improvement discussed above for our original algorithm (to be developed).

= Conclusion
Test
o	What is the best technique?
o	Is any technique good enough to declare the problem solved?
o	What future research do you recommend?

Going forward, we have gotten the intuition that an ensemble will help for at least the action-level policy net, which
also automatically allows for informed variance during control and maybe even different strategies. Then, we also see 
that latent space planning is a promising avenue, which gives our problem formulation some basis. However, we would 
also like to keep the eventual model simplistic without too many engineering tricks to keep it well motivated and 
based in neuroscience. More than model's time-to-performance, we are interested in its design details and 
suitability to scientific ideas, as well as eventual performance and overall interpretability.... Modularization to 
test different hypotheses (adhoc or not), like model-free and model-based robust control,stiffness strategies, etc.
will also be an important factor in the final design. What we hope to achieve is a mildly speculative model of robust
optimal feedback control for a realistic arm, and more importantly, one that will help us simulate various hypotheses
and organization of hierarchical and nested control systems in the nervous system. 

#set heading(numbering: none)
= Notes
Our work (in-progress) on #smallcaps("MotorNet-Pro") can be found in the following repository: 
#link("https://github.com/asadian98/motornet-pro")[#text(fill: rgb("#FF55A3"))[
`https://github.com/asadian98/motornet-pro`]]. We would also like to highlight the brand-new 
#link("https://typst.app/")[#text(fill: rgb("#FF55A3"))[`Typst`]] 
#link("https://github.com/typst/typst")[open-source typesetting system] using which this report 
was gracefully written.

#bibliography("bibliography.bib", title: "References", style: "apa")
