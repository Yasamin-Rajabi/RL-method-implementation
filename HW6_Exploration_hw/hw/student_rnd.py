"""Section C of the homework: Random Network Distillation (RND).

Edit this file through this ``%%writefile hw/student_rnd.py`` cell, then re-run it
and ``importlib.reload(student_rnd)`` (the quick-test cell below) to pick up your
edits without restarting the kernel.

**Action selection is provided** -- RND acts with the same standard epsilon-greedy
behaviour policy as plain DQN, so you implement only the RND-specific learning core
below.

Conventions used throughout this file:

* ``obs`` -- a flattened observation vector; ``obs_size`` is its length.
* ``B`` -- the batch size (number of transitions in a minibatch).
* ``rms`` -- a running mean/std tracker (``common.RunningMeanStd``): an online
  estimator of the mean and standard deviation of a stream of numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from rnd_dqn import RNDDoubleDQN


def build_rnd_networks(
    obs_size: int, output_dim: int, hidden_sizes: tuple[int, ...]
) -> tuple[nn.Module, nn.Module]:
    """Build the two networks RND compares to measure novelty.

    RND scores novelty by how well a *trained predictor* reproduces the output of a
    *fixed random target* on a state: familiar states are easy to predict, novel
    ones are not.

    Args:
        obs_size: length of a flattened observation -- the input width both
            networks must accept (they are called on a ``(B, obs_size)`` tensor).
        output_dim: width of the embedding both networks must emit (their output
            is ``(B, output_dim)``).
        hidden_sizes: a suggested hidden-layer shape, e.g. ``(64, 64)``. Only a
            hint -- you are free to ignore it.

    Returns:
        ``(target, predictor)``, two ``torch.nn.Module`` objects that each map a
        ``(B, obs_size)`` float tensor to a ``(B, output_dim)`` embedding:

        * **target** -- a randomly initialised network that is *never trained*. Its
          parameters must not receive gradients (set ``requires_grad=False`` on
          them) so it stays a fixed, arbitrary function of the state.
        * **predictor** -- a trainable network, regressed toward the target on
          visited states elsewhere in the code.

        Build them however you like -- any architecture works as long as it
        respects the shape contract; a small multi-layer perceptron is the usual
        choice. The two networks need not share an architecture (giving the
        predictor a different, e.g. smaller, capacity is legitimate), but their
        outputs must be the same width so they are directly comparable.
    """
    # TODO(student): frozen random target + trainable predictor, both MLPs obs -> embedding
    def make_mlp():
        layers = []
        in_dim = obs_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    target = make_mlp()
    predictor = make_mlp()

    for param in target.parameters():
        param.requires_grad = False

    return target, predictor


def rnd_intrinsic_bonus(
    agent: "RNDDoubleDQN", next_obs: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Novelty bonus for the landed-in states, and the predictor's training loss.

    Args:
        agent: the RND agent. The pieces you need:

            * ``agent.rnd_target`` -- the frozen target network you built.
            * ``agent.rnd_predictor`` -- the trainable predictor you built.
            * ``agent.intrinsic_rms`` -- a running mean/std tracker for the raw
              novelty scores. Call ``agent.intrinsic_rms.update(arr)`` with a 1-D
              NumPy array to fold a new batch into the statistics, and read
              ``agent.intrinsic_rms.std`` (a float) for the current standard
              deviation.
        next_obs: ``(B, obs_size)`` batch of states the agent just stepped into.

    Returns:
        ``(intrinsic, predictor_loss)``:

        * **intrinsic** -- a ``(B,)`` tensor, one novelty score per state, measuring
          how far the predictor's embedding sits from the target's. It feeds the
          reward, so it must be **detached** from the autograd graph and kept on a
          stable scale: first update ``agent.intrinsic_rms`` with this batch's raw
          scores, then divide the scores by ``agent.intrinsic_rms.std`` (guard
          against division by zero).
        * **predictor_loss** -- a single **differentiable** scalar that, when
          minimised, pulls the predictor toward the frozen target on ``next_obs``.
    """
    # TODO(student): per-state predictor-vs-target error -> differentiable loss + detached rms-normalised bonus
    target_embed = agent.rnd_target(next_obs)
    pred_embed = agent.rnd_predictor(next_obs)

    raw_scores = 0.5 * torch.sum((pred_embed - target_embed) ** 2, dim=-1)

    raw_scores_np = raw_scores.detach().cpu().numpy()
    agent.intrinsic_rms.update(raw_scores_np)

    std = agent.intrinsic_rms.std
    eps = 1e-8
    intrinsic = raw_scores.detach() / (std + eps)

    predictor_loss = torch.mean(raw_scores)

    return intrinsic, predictor_loss


def rnd_td_target(
    agent: "RNDDoubleDQN",
    shaped_reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
) -> "torch.Tensor":
    """The Double-DQN bootstrap target (TD = temporal difference), on the shaped reward.

    Standard Double DQN: the *online* network (``agent.online``) chooses the next
    action, the *target* network (``agent.target``) supplies its value, and a
    terminal transition (``done``) contributes no future value. Discount with
    ``agent.cfg.gamma``; build it under ``torch.no_grad()`` (it is a fixed
    regression target, not something you backpropagate through).

    Args:
        agent: the RND agent (use ``agent.online``, ``agent.target``,
            ``agent.cfg.gamma``).
        shaped_reward: ``(B,)`` reward already combining extrinsic + intrinsic.
        next_obs: ``(B, obs_size)`` next states.
        done: ``(B,)`` float terminal flags (1.0 = terminal, 0.0 = not).

    Returns:
        A ``(B,)`` tensor of bootstrap targets, with no gradient attached.
    """
    # TODO(student): Double-DQN target on shaped_reward
    with torch.no_grad():
        next_q_online = agent.online(next_obs)
        best_actions = torch.argmax(next_q_online, dim=-1)

        next_q_target = agent.target(next_obs)
        next_v = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        gamma = agent.cfg.gamma
        target = shaped_reward + gamma * next_v * (1.0 - done)

    return target


def rnd_q_loss(
    agent: "RNDDoubleDQN",
    obs: "torch.Tensor",
    action: "torch.Tensor",
    shaped_reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
) -> "torch.Tensor":
    """The Double-DQN critic loss for RND, on the shaped reward.

    Take ``Q(s, a)`` from ``agent.online`` for the actions actually taken, build the
    bootstrap target with your :func:`rnd_td_target` on ``shaped_reward``, and
    return the Huber (smooth-L1) loss between them.

    Args:
        agent: the RND agent (use ``agent.online``).
        obs: ``(B, obs_size)`` states.
        action: ``(B,)`` long tensor of actions taken.
        shaped_reward, next_obs, done: as passed to :func:`rnd_td_target`.

    Returns:
        A single differentiable scalar loss.
    """
    # TODO(student): Huber(Q(s,a), rnd_td_target(...)) on the shaped reward
    q_values = agent.online(obs)
    q_selected = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    target = rnd_td_target(agent, shaped_reward, next_obs, done)

    loss = F.smooth_l1_loss(q_selected, target)
    return loss
