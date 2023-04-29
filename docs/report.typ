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
    environmental-knowledge can be thought to be encoded in the form of internal models which are central to
    inferring sensorimotor control. These models can be forumlated various ways, implicitly or explicitly, depending on
    the modelling perspective and level of abstraction (e.g. control-theoretic modules, algorithmic procedures, or
    implicit transformations in artificial neural networks). This work aims to externalize the iterative internal models
    for the control of a realistic arm model by using the model-based reinformcent learning (RL) framework,
    which is contrary to the predominant and performant recurrent neural network (RNN) controllers wherein the
    internal models and control hierarchy are entirely obscured. In this study, we present the integration of a
    realistic and differentiable arm model and then train a baseline model-free stochastic RL policy to control the arm.
    We also propose a model-based RL formulation through which we may be able to ask further scientific questions on
    the organization of naturalistic control as well as achieve better artificial control methods. Altogether, these
    form the first ideas for MotorNet-Probabilistic (or #smallcaps("MotorNet-Pro") in short).
  ],
  keywords: [sensorimotor control, model-based continuous feedback control, stochastic policy],
)

= Introduction
Motor behaviors that may come intuitively to us are not easy to achieve synthetically. The body is a complex system, all
the way from the anatomy -- featuring overactuated and nonlinear musculoskeletal dynamics -- to noisy and delayed
sensorimotor modalities -- with indirect and distinct feedback spaces as well as distributed control structures.
To overcome such complexities, our bodies use internal models to predict and control movements in an adaptive
manner, producing optimal and robust control even in unpredictable and uncertain environments @InternalModels. However,
our model-based control policies, can also display _model-free_ strategic changes to adapt to
unexpected environmental conditions eventhough reflexive reactions to perturbations may be model- and policy-based
#cite("Shadmehr1994", "Conditt1997", "CrevecoeurRobustControl"). This shows the need for a more general framework and
formulation that models the organization of control -- not as a single model-based policy output but one that also allows
for policy modulation, adaptation (e.g. continual learning), and online multi-level planning and control. Added to that, the
framework should be grounded in stochastic control, as this is a key feature of naturalistic movements which also makes them
inherently tunable and robust while also enabling a learning mechanism #cite("SigDepNoise", "VarLearning").

To this end, most neuroscientifically-motivated modelling attempts have included an RNN-based controller, which favorably
lends itself to neuron-population level analyses of activity patterns which are similar to what is done on experimental
recordings from the motor-related cortical areas. MotorNet by @MotorNet #label("motornet") is one such realistic arm
model coupled with a controller that is trained in delayed continuous feedback and continuous action spaces, following
the optimal feedback control (OFC) framework #cite("OFCTodorov", "OFCScott"). This is a promising direction,
especially when studying neural dynamics in cortical control, however, we are interested in the organizational aspect of
motor control in a more _interpretable_ fashion, for which the model-based RL framework is more suitable. In a study by
@RNN-SAC, performant control is elegantly achieved with an RNN-based actor in soft actor-critic (SAC) from @SAC so that
the policy can learn to do online control by maintaining a stateful representation of the ongoing movement and task
demands. However, similar to MotorNet, the purpose of this controller was to investigate proximity of neural activity in
the RNN-pool and with neural recordings of the motor cortex. A study from @BiomechRL also trained a SAC policy, but for a
much simpler arm model, and was focussed on explaining certain movement laws that arise in the learnt stochastic policy
through systematic assumptions in plant modelling, including some that were laid out in @SigDepNoise. This was in an
attempt to eventually propose that our motor system can be understood at a simpler scale without the need for always
caring about the biomechanical complexity. However, in the face of complexities in robot control as well as in the
scientific endeavor to understand how we control our bodies in the first place, we would like to include suitable
complexity in the plant model to gain important and translatable insights for both the engineering and scientific goals
-- which are not too much in isolation as the sensorimotor sciences ("reverse engineering") provide important clues for
robotics ("forward engineering") and vice-versa.

= Problem Formulation
Based on our rationale above, the problem we are interested in is to devise a model-based stochastic RL policy for
controlling a realistic arm model to perform planar reaching. The arm model and the planar reaching task (without the
influence of gravity) form the "environment" in the traditional RL sense. To include the biomechanical complexity of the
arm, we utilize the MotorNet plant in @MotorNet without its controller and auxiliary modules as shown in
@motornet_plant. We created a biologically realistic model with 10 muscles (properties are shown in Figures 2 and 3),
instead of using the 6-muscle default.

#grid(
  columns: 2,
  rows: auto,
  column-gutter: 0em,
  row-gutter: 2em,
  align(center + horizon)[
    #hide[space] // 1em
    #figure(
      image("./assets/plant.svg", width: 80%),
      caption: [#link("motornet")[MotorNet] plant @MotorNet],
    ) <motornet_plant>
  ],
  grid(
    columns: 1,
    rows: (3cm, 3cm),
    column-gutter: 0em,
    row-gutter: 2em,
    figure(image("./assets/momentarms.png", width: 100%), caption: [Moment arms]),
    figure(image("./assets/musclelengths.png", width: 100%), caption: [Muscle lengths]),
  ),
)

A major advantage of relying on MotorNet compared to popular and often simplistic models for RL in MuJoCo, OpenSim, or
MyoSuite is that it additionally allows for differentiability of the arm's musculoskeletal dynamics. This can be
exploited for faster learning or credit assignment in Deep-RL networks instead of only relying on exploration to learn
the same.

In terms of the task, note that here we do not consider bimanual control or include the wrist to consider grasping and
hand orientation control. The single arm model is considered to be a model of the dominant arm for which we want to learn
an interpretable reach control policy. By interpretable, we mean that through the organization, formulation, and state(s)
of the control policy, we would like to be able to understand the control law in terms of the internal models,
reflexive-to-voluntary control and corresponding nested feedback hierarchy that it has learnt. This is in contrast to
the black-box nature of the RNN-based controllers in #cite("MotorNet", "RNN-SAC"), among others. This is a grand goal,
therefore, we discuss some potential approaches in @exploratory_formulation.

Actions in our environment are defined as the activation of each muscle, which results in a 10-dimensional action space.
The state is determined by a variety of observations, including delayed proprioceptive feedback (muscle length and
velocity for each muscle), delayed visual feedback (position of the arm), delay-perceived target feedback, and previous
muscle activations (efferent copy for each muscle), resulting in a 34-dimensional state space. To account for the delay
in information processing in the nervous system, we have set a proprioceptive delay of 20 ms and a visual delay of
70 ms. Additionally, to account for noise in neural pathways, we have added a normal noise with $sigma = 0.01$ to each
feedback state and an normal activation (motor) noise with $sigma = 0.001$. Efferent copies are treated as nondelayed and
noiseless as a minor assumption. The noise model is not signal-dependent, which is another simplification, but would be an
important modelling consideration in future work. The policy rollout lasts for 1 second of simulated time and signifies the
maximum time to reach a target, and the action commands and are updated every 20 ms during the rollout.

We roughly follow the loss definitions in @MotorNet to calculate the reward signal, given below.

$
  L_1 := "mean"_("#joints") (|"goal joint states" - "current joint states"|) \
  L_2 &:= (u_t^tack.b f / ||f||_2^2)^2 \
  L_3 &:= "joint limit cost" \
  L_4 &:= "mean"_("#muscles") (|u_t - u_(t-1)|) \
  L_5 &:= "mean"_("#muscles") ("muscle velocity"^2) \
  r &= - (L_1 + L_2 + L_3 + L_4 + 0.1L_5).
$

Here $L_1$ measures the distance between the desired joint angles and the current joint angles at each time
point. This is necessary to ensure that we are reaching the target accurately. To account for the metabolic cost of
activating muscles, we have included an appropriately scaled muscle activation penalty ($L_2$) in the reward system. This
involves two vectors, $u_t$ and $f$, which represent muscle activations (between 0 and 1) and maximum isometric muscle
forces, respectively. In addition, $L_3$ prevents the arm from moving beyond its natural limits. This is because each
joint angle has a limit beyond which it should not and cannot be controlled. More details on this can be found in the code
implementation. To further improve the reward system, we have included two other terms: $L_4$ to discourage sudden
changes in muscle activation and $L_5$ to limit ongoing effect of previous activations through muscle velocity. It is
worth noting that while the first two terms are used by default in @MotorNet, terms $L_3$, $L_4$, and $L_5$ in the reward
have been designed keeping in mind the nature of RL formulation and training. We use a discount factor $gamma = 0.95$
in our environment. @motornet_pro_layout gives an overview of how the RL model will interact with the MotorNet plant
during and after training.

#align(center)[
  #figure(
    image("./assets/motornet_pro_layout.png", width: 50%),
    caption: [#smallcaps[MotorNet-Pro] layout, adapted from @MotorNet],
  ) <motornet_pro_layout>
]

== Related Work
While more conventional and earlier approaches for continuous control began with the DDPG algorithm @DDPG, many have since
used model-predictive control (MPC) to solve the optimal-control problem with a black-boxed policy. Although MPC-based
policy and value function approximation may provide high explainability about policy behavior and have a broad set of
theoretical tools for formal verification of stability, as described by @PETS, it is not a good fit for our problem due to
its computational expense and inability to meet our criteria for explicit representation of the policy and internal models.
To establish a baseline for comparison and gain insights into the learned policy, we sought compatible implementations of
related papers. Since MotorNet is implemented using TensorFlow, we excluded some model-based RL approaches such as
#cite("MBPO", "MAAC", "PSRL", "SVG", "ModelBasedImplicitDifferentiation") due to their PyTorch or undisclosed
implementations, eventhough they showed promising results. Specifically for reach control, we found a DDPG setup implemented
for attaining a generalized policy by training multiple arms together for different moving targets @idataist. This is a
compelling training approach that we may use in the future. Similar to our environment, this project had a 33-dimensional
observation space albeit with direct joint-angle control and lacking important non-linearities and realistic features such
as noise, delay, and muscle-level control.

There are also image-based control algorithms that infer from a latent state as described by #cite("SLAC", "PlaNet",
"DRQv2"). However, to the best of our knowledge, this literature sufficiently motivated the latent space approach for
embodied, real-life control (e.g. robotics). Furthermore, engineering innovations made in these algorithms (such as
perturbing a serial image frame by $plus.minus 4$ pixels) do not seem to involve any biological detail related to changes in
feedback structure or processing. However, PlaNet @PlaNet makes an important contribution of Recurrent State Space Model
(RSSM) that predicts the future in the latent space by splitting the state into stochastic and deterministic parts, allowing
the model to robustly learn to predict multiple futures. A different but similar model strategy, Dreamer @Dreamer, was
developed that can analytically calculate and exploit gradients from state transitions to speed up the RSSM-model learning.
Adapting these latent-space ideas for the proposed idea in @conceptual_policy is a promising direction for the future.

// Along the topic of planning and control in latent space, work by @ALM on aligning these latent space models each of
// which were previously with their own auxiliary objectives, making the submodel alignment unclear. Single objective
// which jointly optimizes a latent-space model and policy to achieve high returns while remaining self-consistent.
// This is a very close analogue to our bilevel policy formulation, as it a model that predicts in representation space
// for the feedback instead of high-dimensional observation space, and a policy that acts based on those representations.

// This is different than the proposed conceptual formulation @conceptual_policy is the implicit inclusion of online
// planning through continual state prediction in the latent space, to a variable extent conditioning the actor on the
// goal(s) thereby setting the context for action. This naturally falls into the scope of, however, more thought is
// needed on the mechanism and appropriate setup and formulation for adaptive contraction and expansion of feedback
// prediction space. An alternative would be to possible that the transformer and variational autoencoder (VAE) based
// models.

// MBPO: ensemble of models trained using MLE, trained on environment data only and then used to produce cheaper
// rollouts in each episode to train the policy further from the sampled states. (some criticism from MAAC and PSRL).
// When the model class is misspecified or has a limited representational capacity, model parameters with high
// likelihood might not necessarily result in high performance of the agent on a downstream control task
// @ModelBasedImplicitDifferentiation. Nishkin et al. propose a method to address this issue by proposing an end-to-end approach for model learning which directly optimizes the expected returns using implicit differentiation.

== Exploratory Formulation <exploratory_formulation>
Based on the concept of optimal feedback control (OFC) in neuroscience #cite("OFCTodorov", "OFCScott"), it is proposed that
control evolves in time and is online in nature and is based in feedback space, meaning that although we can produce
entirely feedforward control (inverse model) we always use this ability with the short-period feedback prediction (forward
model) to produce the movement. This is supplemented by the longer-term and higher-level planning that is also online and
adaptive, albeit at a slower rate. This is also a biological hierarchy and has recently also been demonstrated in RNNs as an
underlying and necessary mechanism #cite("RNNDynamics", "RNNPrep") for optimal control. Noteably, the forward state
prediction can happen at variable timescales, which makes sense with complex dynamics and changing precision requirements --
which may manifest as chunking of longer sequences of actions or coarser-but-longer prediction horizon over a similar
rollout-timescale, as suggested by @ACT. This hierarchical control structure will require to operate in latent space for
seamless composition whose advantages have recently been demonstrated in a few model-based RL approaches #cite("DRQv2",
"Dreamer", "ALM"). However, variable timescales will require a flexible usage of latents which may introduce complexity, but
it can also allow for a more natural emergence of control strategies depending on the task demands as well as any
environmental perturbations. Summarized in @conceptual_policy, this idea is consistent from many perspectives in
sensorimotor neuroscience, but will need some work in formulating further.

/*
Based on the recent innovation novel algorithm Action Chunking with Transformers (ACT) which reduces the effective
horizon by simply predicting actions in chunks, we would also like that as a generalization of the policy formulation
@ACT. This can also be supported by the neuroscience literature in the sense that a near short horizon plan is known
at the time of execution although the cost landscape for actions beyond the current may evolving in an online manner.
Bringing this formulation into the RL framework can be hard and for now is beyond the scope of this report. But this
is something we will definitely try to incorporate simply in the future.

Similar to how optimal control is done in the brain, with a higher-level context coming in, also recently demonstrated
to be critical to a @RNNPrep. This highlights the need for a plan and also the fits the introspection of having
constructed a general plan or strategy of movement before.

Both forward and inverse models are used in control: policy usually acts as an iterative inverse model that is directed
towards the goal. We would like to fuse the feedforward capability (inv model) to use feedback control (forward model)
to be able to create an online optimal robust controller; for this, the policy should have distilled the knowledge about
the stateprediction and subsequent control -- the dreaming and latent architecture is good for that #cite("SLAC", "PlaNet", "DRQv2", "Dreamer"). dreamer since we are able to think in those models.

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

All of these "snapshots" can be replayed to train an RNN, which will regularize learning of representations and provide
us some control over interpretability of the same -- going from encoded and learnt rules to a universal differential
equation.

Such a control policy can then also be applied to robots and control of other limbs and the whole body.
*/

// todo: why not train 2level policy directly on the latent space, and then how much different would this be from ALM?
// todo: need to see if we need a time-varying time-stepped feedback latent space model; how would we do that??
// should provide intermediate and pre-timept true feedback to train as we are making errors... (non-biological but)
// maybe only briefly mention if we even need this, since over time and experience you would tend to sample inter-
// mediate feedback, only that it won't be timestamped...

#align(center)[
  #figure(
    image("./assets/latent_ofc_policy.png", width: 55%),
    caption: [Latent OFC policy],
  ) <conceptual_policy>
]

= Baseline Algorithm and Considerations
As a starting point, we decided to implement the SAC algorithm for policy optimization. SAC is a popular model-free deep
reinforcement learning algorithm used for continuous control tasks. It is based on the maximum entropy reinforcement
learning framework, which aims to maximize the expected return of a policy while also maximizing the entropy of the
policy distribution. By maximizing the entropy, SAC encourages exploration and prevents the policy from getting stuck in
local optima. At each iteration, SAC performs two main steps: policy evaluation and policy improvement. In the policy
evaluation step, SAC estimates the state-action value function $Q_pi$ which represents the expected return starting from
a given state-action pair under the current policy. The $Q$-function is updated using the Bellman backup operator $Q(s,a)
= EE [r + gamma V(s')]$, where $r$ is the immediate reward, $gamma$ is the discount factor, and $V(s')$ is the value
function of the next state. The policy improvement step involves training a stochastic policy that minimizes the expected
Kullback-Leibler (KL) divergence between the current policy and the exponential of the $Q$-function minus a value
function $V$. The objective function for policy improvement is given by $J(π) = EE [Q_pi(s,a) - alpha log(pi(a|s))]$,
where $alpha$ is a temperature parameter that controls the balance between maximizing the expected return and maximizing
the entropy @SAC. The algorithm is given below for reference.

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
        bold(s)_(t+1))}$\
  #hide("    ")#strong[end for]\
  #hide("    ")#strong[for] each gradient step #strong[do]\
  #hide("    ")#hide("    ") $psi arrow.l psi - lambda_V hat(nabla)_psi J_V(psi)$\
  #hide("    ")#hide("    ") $theta_i arrow.l theta_i - lambda_Q hat(nabla)_theta_i J_Q(theta_i)$ for $i in {1, 2}$\
  #hide("    ")#hide("    ") $phi.alt arrow.l phi.alt - lambda_pi hat(nabla)_phi.alt J_pi(phi.alt)$\
  #hide("    ")#hide("    ") $macron(psi) arrow.l tau psi + (1 - tau)macron(psi)$\
  #hide("    ")#strong[end for]\
  #strong[end for]\
  #v(-0.8em)
  #line(length: 8cm)
  // optional: caption
]

We chose SAC because it is easy to implement and is widely included in performant third-party libraries known to
have shown promising results in a variety of domains. For this purpose, we used the #link("https://www.tensorflow.org/
agents")[TF-Agents] library in TensorFlow and packaged the MotorNet plant inside a custom TensorFlow Environment to make
it compatible with TF-Agents. This was a logical choice given that MotorNet is also a TensorFlow-based library. While we
will be using SAC as our baseline algorithm, we are also considering the approach proposed in REDQ @REDQ for future use.
They have suggested an ensemble of $Q$-function models combined with in-target minimization as an improved method, which we
will consider as a potential optimization, since this forward-engineering insight of using ensemble of value networks also
matches the concept of competing motor plans in sensorimotor neuroscience @CompetingPlans. SAC being a model-free
algorithm, means that it does not require a physical model of the environment. However, it does use a value network which
can be seen as a learned abstraction of the environment's dynamics, detached from its physicality, but nonetheless being
based on an (abstract) model. This still met our rationale of using model-based RL, and hence served as a good baseline
which can be capable of discovering and exploiting important relations in the value-space.

== Preliminary Results
We extensively trained our model over 10,000 episodes to reach the target within one second. @sac_learning_curve
showcases the learning curve, and the best episode rewarded our model with -249.65. Interestingly, we observed optimal
performance after 2000 episodes, but then a gradual decline occurred. This decline may be attributed to "catastrophic
forgetting," where the model's success causes it to forget what failure looks like. As a result, the model predicts high
values for all states and features, regardless of their relevance. When the model encounters unexpected situations with
incorrect predicted values, the error rate can be more and recovery can be challenging. Additionally, the model may
incorrectly link features of the state representation, making it difficult to distinguish between various parts of the
feature space. This creates unusual effects on the model's learning about the values of all states. While the RL model
may behave incorrectly for a few episodes before relearning optimal behavior, it may also break down entirely and never
recover. Here, it is important to note that our model was trained on a single target from a single starting position. As
a result, the policy did not have the need to generalize, which means that running more episodes would have lead to a
large number of similar rollouts -- ultimately resulting in catastrophic forgetting. However, in the case of the original
complex problem of reaching any or different targets from various starting positions, this issue is not likely to arise
quickly or easily.

#align(center)[
  #figure(
    image("./assets/sac_learning_curve.svg", width: 50%),
    caption: [SAC learning curve],
  ) <sac_learning_curve>
]

= Discussion, Future Work, and Conclusion
To get towards a physical model-based baseline algorithm, there are several potential options we can explore. One option is
to include an ad-hoc model by using DynaQ-style @SuttonDyna-Q virtual rollouts based on a continuously-updated physical
model during model-free RL exploration. However, implementing this approach too early in the learning process may not be
beneficial, as it could lead to inaccurate models and incorporating virtual rollouts into the replay buffer may pose
similar challenges. Therefore, a possibility in this direction is to apply the theoretical work on risk-aware model
learning, as demonstrated in PAML @PAML and related works. However, this approach would require some adjustments to suit
the evolving stochastic continuous control framework. Another option is to use a model-based RL algorithms, such as MAAC
and SVG #cite("MAAC", "SVG"), both of which will need custom TensorFlow implementation from scratch.

Looking forward, we have identified the ensemble-$Q$ function approach in REDQ @REDQ to be a potential means to improve the
Deep-RL network, which could enable ensemble-informed stochasticity in control commands and possibly also lay a path for the
emergence of different control strategies in online control. It is worth noting such areas of potential improvement to then
integrate into our conceptual algorithm in @conceptual_policy, which has yet to be developed. We also see latent space
planning as a promising approach that underpins our problem formulation, as discussed in @exploratory_formulation.
Nevertheless, we aim to keep the final model simple, without too many engineering tricks, to ensure it is well-motivated and
grounded in neuroscience. Rather than prioritizing the model's time-to-performance, we are interested in its design details,
suitability to scientific ideas, overall rationale, and its ability to test various hypotheses about the hierarchies of
control systems in the nervous system.

We aspire to develop a model-based RL algorithm that encompasses all the components laid out in @conceptual_policy,
rather than just one, as is the case with other model-based algorithms. Overall, our current approach involves experimental
and conceptual exploration of the "core" baseline framework and devise improvements based on its features. It is worth
noting again that our goal is not only to achieve high performance levels and quick convergence but also to develop a
model-based policy that is interpretable and can provide insights into the underlying control laws. We believe that this
combination of performance and interpretability could serve as a valuable oracle for embedding meaningful representations in
RNN-based controllers, and by itself stand as a model of cortical and trans-cortical computation for movement control,
including long-latency reflexes and optimal and robust voluntary motion.

#set heading(numbering: none)
= Notes
Our work (in-progress) on #smallcaps("MotorNet-Pro") can be found in the following repository:
#link("https://github.com/asadian98/motornet-pro")[#text(fill: rgb("#FF55A3"))[
`https://github.com/asadian98/motornet-pro`]]. We would also like to highlight the brand-new
#link("https://typst.app/")[#text(fill: rgb("#FF55A3"))[`Typst`]]
#link("https://github.com/typst/typst")[open-source typesetting system] using which this report
was gracefully written.

#pagebreak()
#bibliography("bibliography.bib", title: "References", style: "apa")
