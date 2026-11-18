"""Section D of the homework: Bootstrapped DQN with randomized priors.

Edit this file through this ``%%writefile hw/student_bootstrap.py`` cell, then
re-run it and ``importlib.reload(student_bootstrap)`` (the quick-test cell below)
to pick up your edits without restarting the kernel.

Conventions used throughout this file:

* ``obs`` -- a flattened observation vector; ``obs_size`` is its length.
* ``B`` -- the batch size (number of transitions in a minibatch).
* ``K`` -- the ensemble size (number of value heads), i.e. ``agent.cfg.num_heads``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from bootstrap_dqn import BootstrappedDQN

class BootstrappedHead(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, prior_scale: float, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.prior_scale = prior_scale

        def make_mlp():
            layers = []
            in_dim = obs_size
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, num_actions))
            return nn.Sequential(*layers)

        self.trainable = make_mlp()
        self.prior = make_mlp()

        for param in self.prior.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trainable(obs) + self.prior_scale * self.prior(obs)

def build_bootstrap_ensemble(
    obs_size: int,
    num_actions: int,
    num_heads: int,
    prior_scale: float,
    hidden_sizes: tuple[int, ...],
) -> "nn.ModuleList":
    """Build the ensemble of value heads that powers Bootstrapped DQN.

    Bootstrapped DQN explores by *disagreement*: independently initialised heads
    disagree about states the data has not pinned down, and that disagreement drives
    deep exploration. Each head pairs a trainable network with a *frozen randomized
    prior* -- a second, randomly initialised network that is never trained. The head
    outputs their sum::

        Q_k(s) = trainable_k(s) + prior_scale * prior_k(s)

    The frozen prior injects a persistent, per-head source of disagreement that
    ordinary weight initialisation would otherwise wash out as the heads fit the
    same data.

    Args:
        obs_size: length of a flattened observation -- each head's input width
            (heads are called on a ``(B, obs_size)`` tensor).
        num_actions: number of Q-values each head outputs (its output is
            ``(B, num_actions)``).
        num_heads: ``K`` -- how many independent heads to build.
        prior_scale: the weight on each head's frozen prior in the sum above.
        hidden_sizes: a suggested hidden-layer shape, e.g. ``(64, 64)``. Only a
            hint -- you are free to ignore it.

    Returns:
        A ``torch.nn.ModuleList`` of ``num_heads`` heads. Each head is a
        ``torch.nn.Module`` mapping ``(B, obs_size)`` to ``(B, num_actions)`` and
        combining a trainable network with a frozen prior as above. Requirements:

        * The prior's parameters must **not** receive gradients
          (``requires_grad=False``) -- the agent's optimizer trains only the
          parameters that still require gradients, leaving the priors fixed.
        * The heads must be **independent** (each gets its own fresh random init, so
          they start out diverse) but **share the same architecture**, so a target
          copy of the ensemble can mirror it parameter-for-parameter via
          ``load_state_dict``.

        Build the heads however you like as long as they meet this contract.
    """
    # TODO(student): K independent heads, each a trainable net + frozen randomized prior
    heads = []
    for _ in range(num_heads):
        heads.append(BootstrappedHead(obs_size, num_actions, prior_scale, hidden_sizes))
    return nn.ModuleList(heads)


def boot_sample_head(agent: "BootstrappedDQN", rng: "np.random.Generator") -> int:
    """Pick which ensemble head to act with for the next episode.

    This is the posterior-sampling step (approximate Thompson sampling): commit to a
    single head and follow it greedily for a whole episode, which is what makes the
    exploration *deep* (coherent over many steps) instead of dithering.

    Args:
        agent: the Bootstrapped-DQN agent (the ensemble has ``agent.cfg.num_heads``
            heads).
        rng: a NumPy random generator to draw from (use it, not the global RNG, so
            the run is reproducible).

    Returns:
        A single head index, an ``int`` in ``[0, agent.cfg.num_heads)``, drawn
        uniformly at random.
    """
    # TODO(student): uniform head index
    return int(rng.integers(0, agent.cfg.num_heads))


def boot_select_action(agent: "BootstrappedDQN", obs: "np.ndarray", head: int) -> int:
    """Greedy action under one ensemble head -- Bootstrapped DQN's behaviour policy.

    During an episode the agent commits to the single sampled ``head`` and acts
    greedily under it (there is no epsilon-greedy noise).

    Args:
        agent: the Bootstrapped-DQN agent. ``agent.ensemble`` is the
            ``torch.nn.ModuleList`` of heads you built; ``agent.device`` is the
            torch device the heads live on.
        obs: a single ``(obs_size,)`` observation (not a batch).
        head: index of the head to act with.

    Returns:
        The greedy action (an ``int``): the argmax over ``agent.ensemble[head]``'s
        Q-values for ``obs``.
    """
    # TODO(student): greedy argmax under agent.ensemble[head]
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_values = agent.ensemble[head](obs_t)
        action = torch.argmax(q_values, dim=-1).item()
    return int(action)


def boot_eval_action(agent: "BootstrappedDQN", obs: "np.ndarray") -> int:
    """Greedy action for *evaluation*: act on the ensemble's mean Q.

    Training samples one head per episode (that is the exploration); at evaluation
    time you exploit the whole posterior instead -- average the Q-values across every
    head in the ensemble and take the argmax.

    Args:
        agent: the Bootstrapped-DQN agent (use ``agent.ensemble`` and
            ``agent.device``).
        obs: a single ``(obs_size,)`` observation (not a batch).

    Returns:
        The greedy action (an ``int``) under the mean Q-values across all heads.
    """
    # TODO(student): argmax of the mean Q over agent.ensemble
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)

        q_list = [head(obs_t) for head in agent.ensemble]
        mean_q = torch.mean(torch.stack(q_list), dim=0)
        action = torch.argmax(mean_q, dim=-1).item()
    return int(action)


def boot_head_target(
    agent: "BootstrappedDQN",
    online_k: "nn.Module",
    target_k: "nn.Module",
    reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
) -> "torch.Tensor":
    """The Double-DQN bootstrap target for a single ensemble head.

    Same Double-DQN rule as :func:`student_rnd.rnd_td_target`, but per head: this
    head's *online* network ``online_k`` selects the next action and its *target*
    copy ``target_k`` evaluates it; terminal transitions get no future value;
    discount with ``agent.cfg.gamma``; build it under ``torch.no_grad()``.

    Args:
        agent: the Bootstrapped-DQN agent (use ``agent.cfg.gamma``).
        online_k: this head's online network.
        target_k: this head's target copy.
        reward: ``(B,)`` rewards.
        next_obs: ``(B, obs_size)`` next states.
        done: ``(B,)`` float terminal flags (1.0 = terminal, 0.0 = not).

    Returns:
        A ``(B,)`` tensor of bootstrap targets for this head, with no gradient
        attached.
    """
    # TODO(student): per-head Double-DQN target
    with torch.no_grad():

        next_q_online = online_k(next_obs)
        best_actions = torch.argmax(next_q_online, dim=-1)


        next_q_target = target_k(next_obs)
        next_v = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        gamma = agent.cfg.gamma
        target = reward + gamma * next_v * (1.0 - done)

    return target


def boot_head_loss(
    agent: "BootstrappedDQN",
    online_k: "nn.Module",
    target_k: "nn.Module",
    obs: "torch.Tensor",
    action: "torch.Tensor",
    reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
    mask: "torch.Tensor",
) -> "torch.Tensor":
    """The masked Double-DQN loss for one ensemble head.

    For this head: take ``Q(s, a)`` from ``online_k`` for the actions taken, build
    the per-head target with your :func:`boot_head_target`, and return the Huber
    (smooth-L1) loss averaged over **only** the transitions whose bootstrap ``mask``
    bit is set. The caller skips heads with an all-zero mask, so ``mask.sum() > 0``.

    Args:
        agent: the Bootstrapped-DQN agent.
        online_k: this head's online network.
        target_k: this head's target copy.
        obs: ``(B, obs_size)`` states.
        action: ``(B,)`` long tensor of actions taken.
        reward, next_obs, done: as passed to :func:`boot_head_target`.
        mask: ``(B,)`` float tensor of 0/1 bootstrap-mask bits selecting this head's
            transitions.

    Returns:
        A single differentiable scalar: the mean Huber loss over this head's masked
        transitions.
    """
    # TODO(student): masked-mean Huber(Q(s,a), boot_head_target(...)) over this head's transitions
    q_values = online_k(obs)
    q_selected = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    target = boot_head_target(agent, online_k, target_k, reward, next_obs, done)

    huber_loss = F.smooth_l1_loss(q_selected, target, reduction='none')

    masked_loss = huber_loss * mask
    return torch.sum(masked_loss) / (torch.sum(mask) + 1e-8)
