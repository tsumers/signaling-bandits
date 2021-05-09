import pandas as pd
from scipy.special import softmax

class StatelessLiteralListener(object):

    def __init__(self, beta, features, possible_worlds):

        self.beta = beta
        self.features = sorted(features)
        self.possible_worlds = possible_worlds

        self.utterance_belief_states = {}

    def beliefs(self, utterance):
        """Return beliefs over feature values conditioned on a single message, which is a feature-value tuple."""

        # Retrieve cached beliefs if we've already stored them
        resulting_beliefs = self.utterance_belief_states.get(str(utterance))

        if resulting_beliefs is None:
            resulting_beliefs = self.condition_worlds_on_message(self.possible_worlds, utterance)
            self.utterance_belief_states[str(utterance)] = resulting_beliefs

        return resulting_beliefs

    def action_policy(self, utterance, action_context):

        belief_df = self.beliefs(utterance)

        # Estimate rewards, then softmax to get choices.
        action_context['listener_reward'] = self.estimate_rewards(belief_df, action_context)
        action_context['listener_choice_prob'] = softmax(action_context['listener_reward'] * self.beta)

        return action_context

    @classmethod
    def condition_worlds_on_message(cls, possible_worlds, message):
        """Return beliefs over feature values conditioned on a single message, which is a feature-value tuple."""

        worlds = possible_worlds.copy(deep=True)

        # Unpack the message into feature and value
        message_feature = message[0]
        message_value = message[1]

        # Filter for worlds which are consistent with this message
        worlds = worlds[worlds[message_feature] == message_value]

        try:
            worlds["probability"] = 1 / len(worlds)
        except ZeroDivisionError:
            raise ValueError("No worlds are compatible with message {}.".format(message))

        return worlds

    @classmethod
    def estimate_rewards(cls, belief_df, action_context):

        # For compatibility, take a reward vector (in the form of a series) and convert to a one-row dataframe
        if isinstance(belief_df, pd.Series):
            belief_df = pd.DataFrame.from_records([belief_df.to_dict()])

        # Filter action / belief dataframes to feature values and sort them to match ordering
        feature_columns = set(belief_df.columns).intersection(set(action_context.columns))
        feature_columns = sorted(list(feature_columns))

        action_for_choice_df = action_context[feature_columns].reindex(feature_columns, axis=1)
        belief_for_choice_df = belief_df[feature_columns].reindex(feature_columns, axis=1)

        # Calculate rewards for each option, marginalizing over beliefs w.
        rewards = belief_for_choice_df.values @ action_for_choice_df.values.T
        return rewards.mean(axis=0)