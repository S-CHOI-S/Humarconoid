from isaaclab.utils import configclass
from arc_rl.utils import (
    ArcRlOnPolicyRunnerCfg,
    ArcRlPpoActorCriticCfg,
    ArcRlAppoAlgorithmCfg,
    ArcRlMipoAlgorithmCfg,
)


@configclass
class KapexRoughPPORunnerCfg(ArcRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 50
    experiment_name = "kapex_rough"
    empirical_normalization = False
    policy = ArcRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = ArcRlAppoAlgorithmCfg(
        class_name="APPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        actor_learning_rate=1.0e-3,
        critic_learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    policy.init_noise_std = 0.4


@configclass
class KapexFlatPPORunnerCfg(KapexRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.empirical_normalization = True
        self.experiment_name = "kapex_flat"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]


@configclass
class KapexStandPPORunnerCfg(KapexRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.experiment_name = "kapex_stand"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
