from humarcscripts.sse.sse_cfg import SseActorCriticCfg, SseOnPolicyRunnerCfg, SsePpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class KapexRoughSAEPPORunnerCfg(SseOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "kapex_rough_se"
    empirical_normalization = False
    policy = SseActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = SsePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        num_synergies=5,
        sae_hidden_dims=[5],  # 32
    )


@configclass
class KapexFlatSAEPPORunnerCfg(KapexRoughSAEPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "kapex_flat_se"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
