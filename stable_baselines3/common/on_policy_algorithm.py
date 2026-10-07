import sys
import time
import warnings
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
import gymnasium

from gym import spaces as gymspaces
from environments.wrappers import VariBadWrapper

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from gymnasium.wrappers import FlattenObservation

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


# Project ID exposed by the MT10 training environment in main.py.
# This is intentionally not registered as a scalar Gymnasium environment:
# one MT10 worker must correspond to one of the ten benchmark tasks.
METAWORLD_MT10_ENV_ID = "MetaWorldMT10-v0"
METAWORLD_MT10_TASKS = (
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
)
METAWORLD_MT10_NUM_TASKS = len(METAWORLD_MT10_TASKS)



def _match_eval_observation_space(env, model_observation_space):
    """Match raw goal-conditioned eval envs to models trained on flattened observations.

    This is a no-op for existing Box-observation environments. If the raw eval
    environment exposes a Dict observation but the PPO model was constructed
    with a Box observation space (for example via FlattenObservation in main.py),
    apply the same FlattenObservation wrapper to the eval environment.
    """
    if isinstance(env.observation_space, spaces.Dict) and isinstance(model_observation_space, spaces.Box):
        env = FlattenObservation(env)
    return env


def _make_compatible_eval_env(
    env_name,
    seed,
    model_observation_space,
    mt10_task_index=None,
    mt10_benchmark_seed=None,
):
    """Create an internal evaluation env matching the training task/observation space.

    Existing environments preserve the original construction behavior.

    Meta-World MT10 is different from an ordinary Gymnasium scalar environment:
    ``MetaWorldMT10-v0`` is the stable project ID exposed by main.py, while each
    subprocess must actually contain one of the ten official MT10 task environments.
    The MT10 branch below mirrors the benchmark construction used by main.py without
    changing PPO training, rollout collection, checkpointing, or any non-MT10 path.
    """
    if env_name == METAWORLD_MT10_ENV_ID:
        # Local import is required because this function also runs inside spawned
        # SubprocVecEnv workers.
        import metaworld

        if mt10_task_index is None:
            raise ValueError("MT10 evaluation requires mt10_task_index")
        if mt10_task_index < 0 or mt10_task_index >= METAWORLD_MT10_NUM_TASKS:
            raise ValueError(f"Invalid MT10 task index {mt10_task_index}")

        benchmark_seed = seed if mt10_benchmark_seed is None else mt10_benchmark_seed
        benchmark = metaworld.MT10(seed=benchmark_seed)
        benchmark_items = list(benchmark.train_classes.items())
        benchmark_task_names = tuple(name for name, _ in benchmark_items)

        if benchmark_task_names != METAWORLD_MT10_TASKS:
            raise RuntimeError(
                "Installed Meta-World MT10 task order differs from the expected project order: "
                f"{benchmark_task_names}"
            )

        task_name, env_cls = benchmark_items[mt10_task_index]
        tasks = [task for task in benchmark.train_tasks if task.env_name == task_name]

        init_each_env = getattr(metaworld, "_init_each_env", None)
        if init_each_env is None:
            raise RuntimeError(
                "This Meta-World installation does not expose _init_each_env, which is "
                "required to adapt the official MT10 benchmark to SB3 SubprocVecEnv."
            )

        env = init_each_env(
            env_cls=env_cls,
            tasks=tasks,
            seed=benchmark_seed,
            use_one_hot=True,
            env_id=mt10_task_index,
            num_tasks=METAWORLD_MT10_NUM_TASKS,
        )

    elif env_name in {"Meta-World/MT1", "MetaWorldReach-v0"}:
        # Preserve the existing Meta-World Reach compatibility path unchanged.
        import metaworld  # noqa: F401

        env = gymnasium.make(
            "Meta-World/MT1",
            env_name="reach-v3",
            seed=seed,
        )
    else:
        # Preserve the original behavior for every other environment.
        env = gymnasium.make(env_name)

    env = _match_eval_observation_space(env, model_observation_space)
    env.reset(seed=seed)
    return env


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.replay_buffer = ReplayBuffer(
            buffer_size=100_000,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )

    def _actor_param_names(self):
        """
        Actor-only parameter name patterns for SB3 ActorCriticPolicy.
        We exclude value-function parameters to keep the baseline clean.
        """
        return {
            "mlp_extractor.policy_net",
            "action_net",
            "log_std",
        }


    def _apply_param_noise(self, std):
        """
        Apply Gaussian noise to actor parameters only.
        Returns a backup dict: param_name -> original tensor.
        """
        if std <= 0:
            return {}

        backup = {}
        actor_keys = self._actor_param_names()

        with th.no_grad():
            for name, p in self.policy.named_parameters():
                if not p.requires_grad:
                    continue
                if any(k in name for k in actor_keys) and ("value" not in name):
                    backup[name] = p.data.clone()
                    p.add_(th.randn_like(p) * std)

        return backup


    def _restore_param_noise(self, backup):
        """Restore parameters saved by _apply_param_noise."""
        if not backup:
            return

        with th.no_grad():
            for name, p in self.policy.named_parameters():
                if name in backup:
                    p.data.copy_(backup[name])


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        replay_buffer: ReplayBuffer=None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        # backup_params = {}
        # if getattr(self, "use_param_noise", False):
        #     std = float(getattr(self, "param_noise_std", 0.0))
        #     backup_params = self._apply_param_noise(std)


        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # try:

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.reshape(1, -1)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box) or isinstance(self.action_space, gymspaces.box.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if new_obs.ndim != obs_tensor.dim():
                new_obs = new_obs.reshape(obs_tensor.shape)
                self._last_obs = self._last_obs.reshape(obs_tensor.shape)
                rewards, dones, infos = \
                    np.array([rewards]).reshape(values.shape), np.array([dones]).reshape(values.shape), [infos]

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                new_obs,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            if replay_buffer is not None:
                replay_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    new_obs,
                    actions,
                    rewards,
                    dones,
                    infos,
                )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

            if isinstance(env, VariBadWrapper) and np.all(dones):
                self._last_obs = env.reset(seed=self.seed)
            
        # finally:
        #     if backup_params:
        #         self._restore_param_noise(backup_params)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        first_iteration: bool = True,
        init_call: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
            first_iteration
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, replay_buffer=self.replay_buffer)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                # self._dump_logs(iteration)

            self.train()

            if init_call:
                # The training env may have wrappers applied in main.py (for example
                # FlattenObservation for PointMaze/Fetch).  Fresh evaluation envs
                # created here must expose the same observation representation.
                eval_observation_space = self.observation_space

                def make_envs(env_name, seed):
                    def _init(seed_offset):
                        def _thunk():
                            if env_name == METAWORLD_MT10_ENV_ID:
                                # All ten workers come from one MT10 benchmark seed,
                                # while reset seeds retain the existing per-worker offset.
                                return _make_compatible_eval_env(
                                    env_name,
                                    seed + seed_offset,
                                    eval_observation_space,
                                    mt10_task_index=seed_offset % METAWORLD_MT10_NUM_TASKS,
                                    mt10_benchmark_seed=seed,
                                )

                            # Preserve the original evaluation-env construction for
                            # every non-MT10 environment.
                            return _make_compatible_eval_env(
                                env_name,
                                seed + seed_offset,
                                eval_observation_space,
                            )
                        return _thunk
                    return _init
    
                if self.n_envs > 1:
                    # Create a list of environment functions
                    self.env_name = self.env.get_attr("spec")[0].id

                    if self.env_name == METAWORLD_MT10_ENV_ID and self.n_envs != METAWORLD_MT10_NUM_TASKS:
                        raise ValueError(
                            f"MT10 internal evaluation requires {METAWORLD_MT10_NUM_TASKS} environments "
                            f"so every task is represented once; got {self.n_envs}"
                        )

                    print("Creating multiple envs - ", self.n_envs)
                    dummy_env_fns = [make_envs(self.env_name, seed=self.seed)(seed_offset=i) for i in range(self.n_envs)]
                    dummy_env = SubprocVecEnv(dummy_env_fns)
                else:
                    # self.env_name = self.env.spec.id
                    self.env_name = self.env.envs[0].spec.id
                    dummy_env = _make_compatible_eval_env(
                        self.env_name,
                        self.seed,
                        eval_observation_space,
                    )

                # Preserve the original 3-episode evaluation for all existing
                # environments. For MT10, interpret it as 3 episodes per task so
                # all ten tasks are represented uniformly in the diagnostic metric.
                eval_episodes = (
                    3 * METAWORLD_MT10_NUM_TASKS
                    if self.env_name == METAWORLD_MT10_ENV_ID
                    else 3
                )
                returns_trains = evaluate_policy(
                    self,
                    dummy_env,
                    n_eval_episodes=eval_episodes,
                    deterministic=True,
                )[0]
                print(f'avg {eval_episodes} return on policy: {returns_trains}')
                dummy_env.close()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def evaluate(self, num_timesteps):
        # obs = self.env.reset(0)
        obs = self.env.reset(seed=self.seed)
        ret = 0
        for i in range(self.n_steps):
            obs_tensor = obs_as_tensor(obs, self.device).reshape(1, -1)
            actions, _, _ = self.policy(obs_tensor)
            actions = actions.cpu().detach().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box) or isinstance(self.action_space, gymspaces.box.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = self.env.step(clipped_actions)
            if new_obs.ndim != obs_tensor.dim():
                new_obs = new_obs.reshape(obs_tensor.shape)
            obs = new_obs
            ret += rewards
        print(f'Reward at iter {num_timesteps}: {ret}')