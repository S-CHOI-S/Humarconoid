from isaaclab.utils import configclass
from arc_rl.utils import (
    ArcRlOnPolicyRunnerCfg,
    ArcRlPpoActorCriticCfg,
    ArcRlAppoAlgorithmCfg,
    ArcRlMipoAlgorithmCfg,
)


@configclass
class JiwonRoughPPORunnerCfg(ArcRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "jiwon_rough"
    empirical_normalization = True
    policy = ArcRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
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

    # algorithm = ArcRlMipoAlgorithmCfg(
    #     class_name="MIPO",
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.005,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     actor_learning_rate=1.0e-3,
    #     critic_learning_rate=1.0e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )

    policy.init_noise_std = 0.4
    algorithm.value_loss_coef = 0.25
    algorithm.actor_learning_rate = 1.0e-4
    algorithm.critic_learning_rate = 1.0e-4  # 5.0e-4


@configclass
class JiwonFlatPPORunnerCfg(JiwonRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "jiwon_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]

        self.policy.init_noise_std = 0.4
        self.algorithm.actor_learning_rate = 1.0e-4
        self.algorithm.critic_learning_rate = 1.0e-3  # 5.0e-4

        # Auxiliary
        # self.algorithm.auxiliary_cfg = ArcRlAuxiliaryCfg(
        #     input_dim=50,
        #     output_dim=3,
        #     hidden_dims=[32, 32],
        # )

        # Symmetry
        # self.algorithm.symmetry_cfg = ArcRlSymmetryCfg(
        #     use_mirror_loss=True,
        #     mirror_loss_coeff=0.5,  # 추천: 0.2 ~ 1.0 사이
        #     use_data_augmentation=False,
        #     data_augmentation_func=your_symmetry_fn
        # )


@configclass
class JiwonArmPPORunnerCfg(JiwonRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "jiwon_arm"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
