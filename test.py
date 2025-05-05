import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import torch
from stable_baselines3.common.utils import get_latest_run_id
import warnings
import torch.nn as nn
import argparse
from data_collection_config import args_ant
import d3rlpy
from d3rlpy.algos import QLearningAlgoBase
from d3rlpy.base import LearnableConfig
from d3rlpy.constants import ActionSpace
from d3rlpy.torch_utility import TorchMiniBatch, TorchObservation

warnings.filterwarnings("ignore")

device = "cpu"

parser = argparse.ArgumentParser()
args, rest_args = parser.parse_known_args()

env_name = "Ant-v5" # For standard ant locomotion task (single goal task)

if env_name == "Ant-v5":
    args = args_ant.get_args(rest_args)

env = gym.make(env_name) # For Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5

print("Environment created")
print(env.action_space, env.observation_space)
# ------------------------------------------------------------------------------------------------------------

n_steps_per_rollout = args.n_steps_per_rollout

# --------------------------------------------------------------------------------------------------------------

exp = "PPO_test"
DIR = env_name + "/" + exp + "_" + str(get_latest_run_id('logs/'+env_name+"/", exp)+1)
ckp_dir = f'logs/{DIR}/models'

activation_fn_map = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU
}

if hasattr(args, 'use_policy_kwargs') and args.use_policy_kwargs:
    policy_kwargs = {
        "net_arch": [dict(pi=args.pi_layers, vf=args.vf_layers)],
        "activation_fn": activation_fn_map[args.activation_fn]
    }
    if hasattr(args, 'log_std_init'):
        policy_kwargs["log_std_init"] = args.log_std_init
    if hasattr(args, 'ortho_init'):
        policy_kwargs["ortho_init"] = args.ortho_init
else:
    policy_kwargs = None

if hasattr(args, 'use_normalize_kwargs') and args.use_normalize_kwargs:
    normalize_kwargs = {
        "norm_obs": args.norm_obs,
        "norm_reward": args.norm_reward
    }
else:
    normalize_kwargs = None

ppo_kwargs  = dict(
    policy=args.policy,
    env=env,
    verbose=args.verbose,
    seed=args.seed,
    n_steps=args.n_steps_per_rollout,
    batch_size=args.batch_size,
    gamma=args.gamma,
    ent_coef=args.ent_coef,
    learning_rate=args.learning_rate,
    clip_range=args.clip_range,
    max_grad_norm=args.max_grad_norm,
    n_epochs=args.n_epochs,
    gae_lambda=args.gae_lambda,
    vf_coef=args.vf_coef,
    device=args.device,
    tensorboard_log=args.tensorboard_log,
    ckp_dir=ckp_dir
)

if policy_kwargs:
    ppo_kwargs["policy_kwargs"] = policy_kwargs

if normalize_kwargs:
    ppo_kwargs["normalize_kwargs"] = normalize_kwargs

model = PPO(**ppo_kwargs)

# ----------------------------------------------------------------------------------------------------------------

print("Loading Initial saved model")

model.set_parameters(args.init_model_path, device=args.device)

print("Model loaded")

# -------------------------------------------------------------------------------------------------------------

vec_env = model.get_env()
obs = vec_env.reset()

print("Starting evaluation")

# ------------------------------------------------------------------------------------------------------------------

# Define a minimal nn.Module to satisfy base class requirements for _impl
class DummyImpl(torch.nn.Module):
    """
    A minimal placeholder implementation module required by QLearningAlgoBase.
    FQE uses its own implementation but the base class needs this structure.
    Methods here might be called during initialization or by base class logic,
    but FQE's core fitting loop relies on the wrapper's predict_* methods.
    """
    def __init__(self, observation_shape, action_size, device):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.device = device
        # Add a dummy parameter to ensure the module has parameters and can be moved to device
        # This helps avoid potential issues with device placement checks.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        # Return dummy zeros with the expected batch dimension and Q-value shape (batch, 1)
        batch_size = x.shape[0]
        return torch.zeros((batch_size, 1), device=self.device)

    def predict_value(self, x: torch.Tensor, action: torch.Tensor, with_std: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Return dummy values if called directly on impl
        batch_size = x.shape[0]
        zeros = torch.zeros((batch_size, 1), device=self.device)
        if with_std:
            return zeros, zeros
        else:
            return zeros

    def predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        # Return dummy zeros with appropriate action shape.
        batch_size = x.shape[0]
        # For continuous action space, action_size is the dimension
        # For discrete, it would be the number of actions (not handled here)
        action_shape = (batch_size, self.action_size) if isinstance(self.action_size, int) else (batch_size,) + tuple(self.action_size) # Handle tuple shapes
        return torch.zeros(action_shape, device=self.device)

class PPOQWrapper(QLearningAlgoBase):
    def __init__(self, ppo_policy):
        super().__init__(
            config=LearnableConfig(),
            device=str(ppo_policy.device),
            enable_ddp=False
        )
        self.ppo = ppo_policy
        self._action_size = None # To store action size/shape
        self._observation_shape = None # To store observation shape
        self.device = str(ppo_policy.device) # Store device for consistency
        print(f"PPOQWrapper initialized on device: {self.device}")

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS # Default or raise error
        
    # 1. Mandatory method for network initialization
    def inner_create_impl(self, observation_shape: tuple[int, ...], action_size: int | tuple[int, ...]) -> None:
        """ Creates the minimal placeholder _impl required by the base class. """
        print(f"Wrapper inner_create_impl: obs_shape={observation_shape}, action_size={action_size}")
        self._observation_shape = observation_shape
        self._action_size = action_size
        # Create the DummyImpl with the correct shapes and device
        self._impl = DummyImpl(observation_shape, action_size, self.device).to(self.device)
        print(f"DummyImpl created on device: {next(self._impl.parameters()).device}")

    # 2. Required for dataset compatibility
    def build_with_dataset(self, dataset):
        """ Infers shapes from dataset and calls create_impl. """
        if not dataset.episodes:
             raise ValueError("Dataset must contain at least one episode to infer shapes.")

        # Get shapes directly from the dataset object properties
        obs_shape = self.ppo.env.observation_space.shape
        action_shape = self.ppo.env.action_space.shape # This is usually the shape tuple, e.g., (8,)

        # Determine action_size based on action type
        action_type = self.get_action_type()
        if action_type == ActionSpace.CONTINUOUS:
             # For continuous, action_size is often the dimension (first element of shape)
             # Check if action_shape is like (dim,)
             if isinstance(action_shape, tuple) and len(action_shape) == 1:
                  action_size = action_shape[0]
             else:
                  # Handle multi-dimensional continuous actions if necessary
                  action_size = action_shape # Keep the shape tuple
                  print(f"Using action shape tuple as action_size: {action_size}")
        elif action_type == ActionSpace.DISCRETE:
             # For discrete, action_size is the number of possible actions
             action_size = dataset.get_action_size()
             if action_size is None:
                  raise ValueError("Could not determine discrete action size from dataset.")
        else:
             raise ValueError(f"Unsupported action type: {action_type}")
        
        print(f"Building wrapper with: obs_shape={obs_shape}, action_size={action_size} (type: {action_type})")
        # Call create_impl (which calls inner_create_impl)
        self.create_impl(obs_shape, action_size)
        print(f"Wrapper built. Impl set: {self._impl is not None}")

    # 3. Internal helper for prediction (handles numpy/tensor conversion)
    def _predict_sb3(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """ Handles input conversion and calls SB3 PPO predict. """
        if isinstance(x, torch.Tensor):
            # Move tensor to CPU and convert to numpy for SB3
            x_np = x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            x_np = x
        else:
            raise TypeError(f"Unsupported input type for prediction: {type(x)}")

        # Use SB3's prediction (deterministic=True for evaluation)
        action, _ = self.ppo.predict(x_np, deterministic=True)
        return action
        
    # 4. Method FQE calls to get policy actions a' = pi(s')
    @torch.no_grad() # Ensure no gradients are computed during prediction
    def predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        """ Predicts the best action using the PPO policy (deterministic). """
        print("This is the predict_best_action method")
        # Use the helper to get numpy action
        np_actions = self._predict_sb3(x)
        # Convert result back to tensor on the correct device for d3rlpy
        return torch.tensor(np_actions, dtype=torch.float32).to(self.device)
    
    # 5. Method FQE calls to get value estimates V(s') for targets
    @torch.no_grad() # Ensure no gradients are computed during value prediction
    def predict_value(self,
                      x: torch.Tensor,
                      action: torch.Tensor | None = None, # Action ignored for PPO critic V(s)
                      with_std: bool = False
                      ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """ Predicts the state value V(s) using the PPO critic. """
        # Ensure input is a tensor on the correct device
        if isinstance(x, np.ndarray):
            # Convert numpy to tensor if needed (unlikely if called from d3rlpy)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        elif x.device != self.device:
             # Move tensor to the correct device if it's not already there
             x_tensor = x.to(self.device)
        else:
            x_tensor = x

        # Access the critic/value network within the SB3 policy object
        # Common attribute names are 'value_net' or 'vf_net'. Check your SB3 version/policy.
        if hasattr(self.ppo.policy, 'value_net') and self.ppo.policy.value_net is not None:
            values = self.ppo.policy.value_net(x_tensor)
        elif hasattr(self.ppo.policy, 'vf_net') and self.ppo.policy.vf_net is not None: # Another possible name
             values = self.ppo.policy.vf_net(x_tensor)
        else:
             # You might need to inspect self.ppo.policy to find the correct attribute
             raise AttributeError("Could not find the value network (e.g., 'value_net', 'vf_net') in the PPO policy object. Please verify the attribute name.")

        # Ensure output shape is (batch_size, 1) as expected by d3rlpy
        values = values.reshape(-1, 1)

        if with_std:
            # Standard PPO critic doesn't output standard deviation
            # Return zeros as placeholder
            std = torch.zeros_like(values)
            return values, std
        else:
            return values
        
    # --- Other necessary overrides ---

    def save_model(self, fname: str) -> None:
        """ Saving the wrapper itself is not standard. Save the SB3 model separately. """
        print(f"PPOQWrapper.save_model called: Skipping wrapper save. Save the original SB3 PPO model using ppo.save('{fname}') instead.")
        pass

    def load_model(self, fname: str) -> None:
        """ Loading into the wrapper is not standard. Load the SB3 model before creating the wrapper. """
        print(f"PPOQWrapper.load_model called: Skipping wrapper load. Load the SB3 PPO model using PPO.load('{fname}') and then create the wrapper.")
        pass

    def update(self, batch: TorchMiniBatch) -> dict[str, float]:
        """ FQE handles its own updates. This wrapper should not perform updates. """
        # Return empty dict to satisfy the base class method signature
        return {}    

class PPOQWrapper1(QLearningAlgoBase):
    def __init__(self, ppo_policy):
        super().__init__(config=LearnableConfig(), device=str(ppo_policy.device), enable_ddp=False)
        self.ppo = ppo_policy
        self._action_space = self.get_action_type()  # Explicitly set action space
        self.device = str(ppo_policy.device) # Store device for consistency
        print(f"PPOQWrapper initialized on device: {self.device}")

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS # Default or raise error

    # --- Critical Overrides ---
    @torch.no_grad()
    def predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        """Directly use PPO's policy without relying on _impl."""
        if isinstance(x, (list, tuple)):  # Handle complex observations
            x = [xi.to(self.device) for xi in x]
        else:
            x = x.to(self.device)
        
        # Convert to numpy for SB3 compatibility
        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        actions, _ = self.ppo.predict(x_np, deterministic=True)
        return torch.as_tensor(actions, device=self.device)

    @torch.no_grad()
    def predict_value(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        """Directly access PPO's critic network."""
        # Use the critic network as in your original code
        if hasattr(self.ppo.policy, 'value_net'):
            return self.ppo.policy.value_net(x)
        else:
            raise AttributeError("PPO critic network not found")

    # --- Required Base Class Methods ---
    def inner_create_impl(self, observation_shape, action_size):
        # Directly use PPO networks instead of dummy
        self._impl = self  # Bypass d3rlpy's impl requirement

    def update(self, batch: TorchMiniBatch) -> dict:
        """No-op since PPO isn't being trained."""
        return {}

# load from HDF5
with open("ppo_ant_d3rlpy_buffer.h5", "rb") as f:
    dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())

# 3. Create the FQE-compatible wrapper
print("\nCreating PPOQWrapper...")
try:
    ppo_wrapper = PPOQWrapper1(model)

    # 4. Build the wrapper internals using dataset info
    print("Building wrapper with dataset...")
    ppo_wrapper.build_with_dataset(dataset)

except Exception as e:
    print(f"Error creating or building the PPOQWrapper: {e}")
    import traceback
    traceback.print_exc()
    exit()

print("--------------------------------------------------------------------------------")

# 5. Configure and Run FQE
print("\nConfiguring FQE...")
try:
    fqe = d3rlpy.ope.FQE(
        algo=ppo_wrapper, # Pass the wrapper instance
        config=d3rlpy.ope.FQEConfig(
            learning_rate=3e-4,         # Learning rate for FQE's internal Q-network
            target_update_interval=100, # How often to update FQE's target network
        )
    )

    print("--------------------------------------------------------------------------------")
    print("Fitting d3rlpy FQE...")

    # Consider using a smaller number of steps for initial testing
    N_STEPS = 500 # 10000
    N_STEPS_PER_EPOCH = 100 # 1000

    output = fqe.fit(
        dataset,
        n_steps=N_STEPS,
        n_steps_per_epoch=N_STEPS_PER_EPOCH,
        evaluators={
            # Estimates the expected value of the initial states according to FQE's learned Q-function
            'init_value': d3rlpy.metrics.InitialStateValueEstimationEvaluator(),
            # Soft Off-Policy Classification: Measures if the policy achieves a certain return threshold
            'soft_opc': d3rlpy.metrics.SoftOPCEvaluator(return_threshold=-300), # Adjust threshold based on env/task
        },
        show_progress=True,
    )

    print("\nFQE Fitting completed.")
    # output contains the metrics from the last epoch
    print("Final Epoch Metrics:", output)

    # The primary result is often the initial state value estimate
    initial_state_value = output[-1][1]['init_value'] # Get from the last epoch's results
    print(f"Estimated Initial State Value: {initial_state_value}")

    # You can also access the soft OPC result if needed
    soft_opc_result = output[-1][1]['soft_opc']
    print(f"Soft OPC Result: {soft_opc_result}")

    # Rank the results based on the initial state value estimate
    ranked_results = sorted(output, key=lambda x: x[1]['init_value'], reverse=True)
    print("Ranked Results (by Initial State Value):")
    for rank, (epoch, metrics) in enumerate(ranked_results, start=1):
        print(f"Rank {rank}: Epoch {epoch}, Initial State Value: {metrics['init_value']}")

    # Rank the results based on the soft OPC result
    ranked_results_opc = sorted(output, key=lambda x: x[1]['soft_opc'], reverse=True)
    print("Ranked Results (by Soft OPC):")
    for rank, (epoch, metrics) in enumerate(ranked_results_opc, start=1):
        print(f"Rank {rank}: Epoch {epoch}, Soft OPC: {metrics['soft_opc']}")

except Exception as e:
    print(f"Error during FQE configuration or fitting: {e}")
    import traceback
    traceback.print_exc()
