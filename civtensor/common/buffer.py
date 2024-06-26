import numpy as np
import torch

from civtensor.utils.trans_tools import _flatten


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class Buffer:
    def __init__(self, args, observation_spaces, action_spaces):
        # init parameters
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.rnn_hidden_dim = args["rnn_hidden_dim"]
        self.n_rnn_layers = args["n_rnn_layers"]
        self.gamma = args["gamma"]
        self.gae_lambda = args["gae_lambda"]
        self.use_gae = args["use_gae"]
        self.use_proper_time_limits = args["use_proper_time_limits"]

        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        # obtain input dimensions. TODO: be consistent with env
        self.rules_dim = self.observation_spaces["rules"].shape[0]
        self.player_dim = self.observation_spaces["player"].shape[0]
        self.token_dim = self.observation_spaces['token'].shape[0]
        self.token_embed_dim = self.observation_spaces["token_embed"].shape[0]
        self.others_player_dim = self.observation_spaces["others_player"].shape[
            1
        ]  # or Sequence?
        self.unit_dim = self.observation_spaces["unit"].shape[1]  # or Sequence?
        self.city_dim = self.observation_spaces["city"].shape[1]  # or Sequence?
        self.others_unit_dim = self.observation_spaces["others_unit"].shape[
            1
        ]  # or Sequence?
        self.others_city_dim = self.observation_spaces["others_city"].shape[
            1
        ]  # or Sequence?
        self.dipl_dim = self.observation_spaces["dipl"].shape[
            1
        ]  # or Sequence?
        self.map_dim = self.observation_spaces["map"].shape
        self.xsize, self.ysize, self.map_channels = self.map_dim
        self.n_max_others_player = self.observation_spaces["others_player"].shape[0]
        self.n_max_unit = self.observation_spaces["unit"].shape[0]
        self.n_max_city = self.observation_spaces["city"].shape[0]
        self.n_max_others_unit = self.observation_spaces["others_unit"].shape[0]
        self.n_max_others_city = self.observation_spaces["others_city"].shape[0]
        self.n_max_dipl = self.observation_spaces["dipl"].shape[0]

        # obtain output dimensions. TODO: be consistent with env
        self.actor_type_dim = self.action_spaces["actor_type"].n
        self.city_action_type_dim = self.action_spaces["city_action_type"].n
        self.unit_action_type_dim = self.action_spaces["unit_action_type"].n
        self.gov_action_type_dim = self.action_spaces["gov_action_type"].n
        self.dipl_action_type_dim = self.action_spaces["dipl_action_type"].n
        self.tech_action_type_dim = self.action_spaces["tech_action_type"].n

        # init buffers
        self.rules_input = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.rules_dim),
            dtype=np.float32,
        )
        self.token_input = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.token_dim),
            dtype=np.float32,
        )
        self.token_embed_input = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.token_embed_dim),
            dtype=np.float32,
        )
        self.player_input = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.player_dim),
            dtype=np.float32,
        )
        self.others_player_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_player,
                self.others_player_dim,
            ),
            dtype=np.float32,
        )
        self.unit_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_unit,
                self.unit_dim,
            ),
            dtype=np.float32,
        )
        self.city_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_city,
                self.city_dim,
            ),
            dtype=np.float32,
        )
        self.others_unit_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_unit,
                self.others_unit_dim,
            ),
            dtype=np.float32,
        )
        self.others_city_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_city,
                self.others_city_dim,
            ),
            dtype=np.float32,
        )
        self.dipl_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_dipl,
                self.dipl_dim,
            ),
            dtype=np.float32,
        )

        self.map_input = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *self.map_dim),
            dtype=np.float32,
        )

        self.others_player_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_player,
                1,
            ),
            dtype=np.int64,
        )
        self.unit_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_unit, 1),
            dtype=np.int64,
        )
        self.city_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_city, 1),
            dtype=np.int64,
        )
        self.others_unit_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_unit,
                1,
            ),
            dtype=np.int64,
        )
        self.others_city_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_city,
                1,
            ),
            dtype=np.int64,
        )

        self.rnn_hidden_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_rnn_layers,
                self.rnn_hidden_dim,
            ),
            dtype=np.float32,
        )

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )

        self.actor_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )
        self.actor_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.actor_type_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.actor_type_dim),
            dtype=np.int64,
        )

        self.city_id_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.int64
        )
        self.city_id_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.city_id_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_city, 1),
            dtype=np.int64,
        )

        self.city_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )  # TODO: check data type
        self.city_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.city_action_type_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_city,
                self.city_action_type_dim,
            ),
            dtype=np.int64,
        )

        self.unit_id_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.int64
        )
        self.unit_id_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.unit_id_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_unit, 1),
            dtype=np.int64,
        )

        self.unit_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )
        self.unit_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.unit_action_type_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_unit,
                self.unit_action_type_dim,
            ),
            dtype=np.int64,
        )

        self.dipl_id_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.int64
        )
        self.dipl_id_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.dipl_id_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_dipl, 1),
            dtype=np.int64,
        )

        self.dipl_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )
        self.dipl_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.dipl_action_type_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_dipl,
                self.dipl_action_type_dim,
            ),
            dtype=np.int64,
        )

        self.gov_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )
        self.gov_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.gov_action_type_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.gov_action_type_dim),
            dtype=np.int64,
        )

        self.tech_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )
        self.tech_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.tech_action_type_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.tech_action_type_dim),
            dtype=np.int64,
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.int64
        )
        self.bad_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.int64
        )

        self.step = 0

    def get_mean_rewards(self):
        return np.mean(self.rewards)

    def insert(self, data):
        """Insert data into buffer."""
        (
            token,
            token_embed,
            rules,
            player,
            others_player,
            unit,
            city,
            dipl,
            others_unit,
            others_city,
            map,
            others_player_mask,
            unit_mask,
            city_mask,
            others_unit_mask,
            others_city_mask,
            rnn_hidden_state,
            actor_type,
            actor_type_log_prob,
            actor_type_mask,
            city_id,
            city_id_log_prob,
            city_id_mask,
            city_action_type,
            city_action_type_log_prob,
            city_action_type_mask,
            unit_id,
            unit_id_log_prob,
            unit_id_mask,
            unit_action_type,
            unit_action_type_log_prob,
            unit_action_type_mask,
            dipl_id,
            dipl_id_log_prob,
            dipl_id_mask,
            dipl_action_type,
            dipl_action_type_log_prob,
            dipl_action_type_mask,
            gov_action_type,
            gov_action_type_log_prob,
            gov_action_type_mask,
            tech_action_type,
            tech_action_type_log_prob,
            tech_action_type_mask,
            mask,
            bad_mask,
            reward,
            value_pred,
        ) = data

        self.token_input[self.step + 1] = token.copy()
        self.token_embed_input[self.step + 1] = token_embed.copy()
        self.rules_input[self.step + 1] = rules.copy()
        self.player_input[self.step + 1] = player.copy()
        self.others_player_input[self.step + 1] = others_player.copy()
        self.unit_input[self.step + 1] = unit.copy()
        self.city_input[self.step + 1] = city.copy()
        self.dipl_input[self.step + 1] = dipl.copy()
        self.others_unit_input[self.step + 1] = others_unit.copy()
        self.others_city_input[self.step + 1] = others_city.copy()
        self.map_input[self.step + 1] = map.copy()
        self.others_player_masks[self.step + 1] = others_player_mask.copy()
        self.unit_masks[self.step + 1] = unit_mask.copy()
        self.city_masks[self.step + 1] = city_mask.copy()
        self.others_unit_masks[self.step + 1] = others_unit_mask.copy()
        self.others_city_masks[self.step + 1] = others_city_mask.copy()
        self.rnn_hidden_states[self.step + 1] = rnn_hidden_state.copy()
        self.actor_type_output[self.step] = actor_type.copy()
        self.actor_type_log_probs[self.step] = actor_type_log_prob.copy()
        self.actor_type_masks[self.step + 1] = actor_type_mask.copy()
        self.city_id_output[self.step] = city_id.copy()
        self.city_id_log_probs[self.step] = city_id_log_prob.copy()
        self.city_id_masks[self.step + 1] = city_id_mask.copy()
        self.city_action_type_output[self.step] = city_action_type.copy()
        self.city_action_type_log_probs[self.step] = city_action_type_log_prob.copy()
        self.city_action_type_masks[self.step + 1] = city_action_type_mask.copy()
        self.unit_id_output[self.step] = unit_id.copy()
        self.unit_id_log_probs[self.step] = unit_id_log_prob.copy()
        self.unit_id_masks[self.step + 1] = unit_id_mask.copy()
        self.unit_action_type_output[self.step] = unit_action_type.copy()
        self.unit_action_type_log_probs[self.step] = unit_action_type_log_prob.copy()
        self.unit_action_type_masks[self.step + 1] = unit_action_type_mask.copy()
        self.dipl_id_output[self.step] = dipl_id.copy()
        self.dipl_id_log_probs[self.step] = dipl_id_log_prob.copy()
        self.dipl_id_masks[self.step + 1] = dipl_id_mask.copy()
        self.dipl_action_type_output[self.step] = dipl_action_type.copy()
        self.dipl_action_type_log_probs[self.step] = dipl_action_type_log_prob.copy()
        self.dipl_action_type_masks[self.step + 1] = dipl_action_type_mask.copy()
        self.gov_action_type_output[self.step] = gov_action_type.copy()
        self.gov_action_type_log_probs[self.step] = gov_action_type_log_prob.copy()
        self.gov_action_type_masks[self.step + 1] = gov_action_type_mask.copy()
        self.tech_action_type_output[self.step] = tech_action_type.copy()
        self.tech_action_type_log_probs[self.step] = tech_action_type_log_prob.copy()
        self.tech_action_type_masks[self.step + 1] = tech_action_type_mask.copy()
        self.masks[self.step + 1] = mask.copy()
        self.bad_masks[self.step + 1] = bad_mask.copy()
        self.rewards[self.step] = reward.copy()
        self.value_preds[self.step] = value_pred.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.token_input[0] = self.token_input[-1].copy()
        self.token_embed_input[0] = self.token_embed_input[-1].copy()
        self.rules_input[0] = self.rules_input[-1].copy()
        self.player_input[0] = self.player_input[-1].copy()
        self.others_player_input[0] = self.others_player_input[-1].copy()
        self.unit_input[0] = self.unit_input[-1].copy()
        self.city_input[0] = self.city_input[-1].copy()
        self.dipl_input[0] = self.dipl_input[-1].copy()
        self.others_unit_input[0] = self.others_unit_input[-1].copy()
        self.others_city_input[0] = self.others_city_input[-1].copy()
        self.map_input[0] = self.map_input[-1].copy()
        self.others_player_masks[0] = self.others_player_masks[-1].copy()
        self.unit_masks[0] = self.unit_masks[-1].copy()
        self.city_masks[0] = self.city_masks[-1].copy()
        self.others_unit_masks[0] = self.others_unit_masks[-1].copy()
        self.others_city_masks[0] = self.others_city_masks[-1].copy()
        self.rnn_hidden_states[0] = self.rnn_hidden_states[-1].copy()
        self.actor_type_masks[0] = self.actor_type_masks[-1].copy()
        self.city_id_masks[0] = self.city_id_masks[-1].copy()
        self.city_action_type_masks[0] = self.city_action_type_masks[-1].copy()
        self.unit_id_masks[0] = self.unit_id_masks[-1].copy()
        self.unit_action_type_masks[0] = self.unit_action_type_masks[-1].copy()
        self.dipl_id_masks[0] = self.dipl_id_masks[-1].copy()
        self.dipl_action_type_masks[0] = self.dipl_action_type_masks[-1].copy()
        self.gov_action_type_masks[0] = self.gov_action_type_masks[-1].copy()
        self.tech_action_type_masks[0] = self.tech_action_type_masks[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """Compute returns either as discounted sum of rewards, or using GAE.
        Args:
            next_value: (np.ndarray) value predictions for the step after the last episode step.
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
        """
        if (
            self.use_proper_time_limits
        ):  # consider the difference between truncation and termination
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:  # do not consider the difference between truncation and termination, i.e. all done episodes are terminated
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.n_rollout_threads
        assert n_rollout_threads >= num_mini_batch, (
            f"The number of processes ({n_rollout_threads}) "
            f"has to be greater than or equal to the number of "
            f"mini batches ({num_mini_batch})."
        )
        num_envs_per_batch = n_rollout_threads // num_mini_batch

        # shuffle indices
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        # prepare data for each mini batch
        for batch_id in range(num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            token_batch = _flatten(T, N , self.token_input[:-1, ids])
            token_embed_batch = _flatten(T, N , self.token_embed_input[:-1, ids])
            rules_batch = _flatten(T, N, self.rules_input[:-1, ids])
            player_batch = _flatten(T, N, self.player_input[:-1, ids])
            others_player_batch = _flatten(T, N, self.others_player_input[:-1, ids])
            unit_batch = _flatten(T, N, self.unit_input[:-1, ids])
            city_batch = _flatten(T, N, self.city_input[:-1, ids])
            dipl_batch = _flatten(T, N, self.dipl_input[:-1, ids])
            others_unit_batch = _flatten(T, N, self.others_unit_input[:-1, ids])
            others_city_batch = _flatten(T, N, self.others_city_input[:-1, ids])
            map_batch = _flatten(T, N, self.map_input[:-1, ids])
            others_player_masks_batch = _flatten(
                T, N, self.others_player_masks[:-1, ids]
            )
            unit_masks_batch = _flatten(T, N, self.unit_masks[:-1, ids])
            city_masks_batch = _flatten(T, N, self.city_masks[:-1, ids])
            others_unit_masks_batch = _flatten(T, N, self.others_unit_masks[:-1, ids])
            others_city_masks_batch = _flatten(T, N, self.others_city_masks[:-1, ids])
            rnn_hidden_states_batch = self.rnn_hidden_states[0:1, ids]
            old_value_preds_batch = _flatten(T, N, self.value_preds[:-1, ids])
            return_batch = _flatten(T, N, self.returns[:-1, ids])
            adv_targ = _flatten(T, N, advantages[:, ids])
            actor_type_batch = _flatten(T, N, self.actor_type_output[:, ids])
            old_actor_type_log_probs_batch = _flatten(
                T, N, self.actor_type_log_probs[:, ids]
            )
            actor_type_masks_batch = _flatten(T, N, self.actor_type_masks[:-1, ids])
            city_id_batch = _flatten(T, N, self.city_id_output[:, ids])
            old_city_id_log_probs_batch = _flatten(T, N, self.city_id_log_probs[:, ids])
            city_id_masks_batch = _flatten(T, N, self.city_id_masks[:-1, ids])
            city_action_type_batch = _flatten(
                T, N, self.city_action_type_output[:, ids]
            )
            old_city_action_type_log_probs_batch = _flatten(
                T, N, self.city_action_type_log_probs[:, ids]
            )
            city_action_type_masks_batch = _flatten(
                T, N, self.city_action_type_masks[:-1, ids]
            )
            unit_id_batch = _flatten(T, N, self.unit_id_output[:, ids])
            old_unit_id_log_probs_batch = _flatten(T, N, self.unit_id_log_probs[:, ids])
            unit_id_masks_batch = _flatten(T, N, self.unit_id_masks[:-1, ids])
            unit_action_type_batch = _flatten(
                T, N, self.unit_action_type_output[:, ids]
            )
            old_unit_action_type_log_probs_batch = _flatten(
                T, N, self.unit_action_type_log_probs[:, ids]
            )
            unit_action_type_masks_batch = _flatten(
                T, N, self.unit_action_type_masks[:-1, ids]
            )
            dipl_id_batch = _flatten(T, N, self.dipl_id_output[:, ids])
            old_dipl_id_log_probs_batch = _flatten(T, N, self.dipl_id_log_probs[:, ids])
            dipl_id_masks_batch = _flatten(T, N, self.dipl_id_masks[:-1, ids])
            dipl_action_type_batch = _flatten(
                T, N, self.dipl_action_type_output[:, ids]
            )
            old_dipl_action_type_log_probs_batch = _flatten(
                T, N, self.dipl_action_type_log_probs[:, ids]
            )
            dipl_action_type_masks_batch = _flatten(
                T, N, self.dipl_action_type_masks[:-1, ids]
            )
            gov_action_type_batch = _flatten(T, N, self.gov_action_type_output[:, ids])
            old_gov_action_type_log_probs_batch = _flatten(
                T, N, self.gov_action_type_log_probs[:, ids]
            )
            gov_action_type_masks_batch = _flatten(
                T, N, self.gov_action_type_masks[:-1, ids]
            )
            tech_action_type_batch = _flatten(T, N, self.tech_action_type_output[:, ids])
            old_tech_action_type_log_probs_batch = _flatten(
                T, N, self.tech_action_type_log_probs[:, ids]
            )
            tech_action_type_masks_batch = _flatten(
                T, N, self.tech_action_type_masks[:-1, ids]
            )
            masks_batch = _flatten(T, N, self.masks[:-1, ids])
            bad_masks_batch = _flatten(T, N, self.bad_masks[:-1, ids])

            rnn_hidden_states_batch = rnn_hidden_states_batch.squeeze(0)

            yield (
                token_batch,
                token_embed_batch,
                rules_batch,
                player_batch,
                others_player_batch,
                unit_batch,
                city_batch,
                dipl_batch,
                others_unit_batch,
                others_city_batch,
                map_batch,
                others_player_masks_batch,
                unit_masks_batch,
                city_masks_batch,
                others_unit_masks_batch,
                others_city_masks_batch,
                rnn_hidden_states_batch,
                old_value_preds_batch,
                return_batch,
                adv_targ,
                actor_type_batch,
                old_actor_type_log_probs_batch,
                actor_type_masks_batch,
                city_id_batch,
                old_city_id_log_probs_batch,
                city_id_masks_batch,
                city_action_type_batch,
                old_city_action_type_log_probs_batch,
                city_action_type_masks_batch,
                unit_id_batch,
                old_unit_id_log_probs_batch,
                unit_id_masks_batch,
                unit_action_type_batch,
                old_unit_action_type_log_probs_batch,
                unit_action_type_masks_batch,
                dipl_id_batch,
                old_dipl_id_log_probs_batch,
                dipl_id_masks_batch,
                dipl_action_type_batch,
                old_dipl_action_type_log_probs_batch,
                dipl_action_type_masks_batch,
                gov_action_type_batch,
                old_gov_action_type_log_probs_batch,
                gov_action_type_masks_batch,
                tech_action_type_batch,
                old_tech_action_type_log_probs_batch,
                tech_action_type_masks_batch,
                masks_batch,
                bad_masks_batch,
            )
