from isaaclab.utils import configclass
from arc_rl.utils import ArcRlOnPolicyRunnerCfg, ArcRlPpoActorCriticCfg, ArcRlAppoAlgorithmCfg


@configclass
class JiwonRoughPPORunnerCfg(ArcRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "jiwon_rough"
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
        critic_learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class JiwonFlatPPORunnerCfg(JiwonRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "jiwon_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]

        self.policy.init_noise_std = 0.2
        self.algorithm.actor_learning_rate = 1e-7


@configclass
class JiwonArmPPORunnerCfg(JiwonRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "jiwon_arm"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
