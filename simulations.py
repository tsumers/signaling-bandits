import itertools
import pandas as pd
from scipy.special import softmax


def generate_actions_from_features(*features):
    """Return a list of all possible shape-color pairs."""

    return list(itertools.product(*features))


def generate_context_from_actions(action_tuples):
    """Given a list of action-tuples, return a dataframe where columns are features of those actions."""

    action_records = []
    action_names = []
    for action in action_tuples:
        action_records.append({f: 1 for f in action})
        action_names.append(" ".join([f for f in action]))

    df = pd.DataFrame.from_records(action_records, index=action_names)

    return df.fillna(0)


def generate_worlds_from_feature_values(features, possible_values):
    """Given lists of features and values, generate a dataframe with all possible feature-value combinations."""

    worlds_list = []

    for f in features:
        if not worlds_list:
            worlds_list = [{f: k} for k in possible_values]
        else:
            worlds_list = [dict(**w, **{f: k}) for w in worlds_list for k in possible_values]

    return pd.DataFrame.from_records(worlds_list)


def speakers_utterance_probabilities_single_action_context(action_context, speaker_list, utterance_list, w):
    """Given an action context, return speaker probabilities and outcomes for each utterance."""

    utterance_results = []
    for u in utterance_list:

        truthful_utterance = bool(w[u[0]] == u[1])

        # Basic utterance information: contents, truthfulness
        individual_utterance = {"utterance": u,
                                "feature": u[0],
                                "value": u[1],
                                "truthful": truthful_utterance,
                                "expected_rewards": speaker_list[0].expected_rewards(u, action_context),
                                "prob_optimal_action": speaker_list[0].probability_optimal_action(u, action_context)}

        for speaker in speaker_list:
            individual_utterance[speaker.name] = speaker.utility(u, action_context)

        utterance_results.append(individual_utterance)

    utterance_utility_dataframe = pd.DataFrame.from_records(utterance_results)
    for s in speaker_list:
        speaker_name = s.name
        utterance_utility_dataframe[speaker_name + "_prob"] = softmax(utterance_utility_dataframe[s.name] * s.beta)

    return utterance_utility_dataframe


def speaker_utterance_probabilities_multiple_action_contexts(action_context_tuples, speaker_list, utterance_list, w):
    """Given a world configuration, generate speaker utterance probabilities over action sets."""

    # Iterate over all possible action sets and return results
    all_results = []
    for action_tuples in action_context_tuples:
        action_context = generate_context_from_actions(action_tuples)
        action_tuple_results = speakers_utterance_probabilities_single_action_context(action_context, speaker_list,
                                                                                      utterance_list, w)
        action_tuple_results.loc[:, "action_context"] = str(action_context.index.values)

        all_results.append(action_tuple_results)

    results_df = pd.concat(all_results)

    return results_df


def summarize_speakers_performance(speaker_action_df, speaker_list):
    speaker_bandit_list = []

    if "action_context" not in speaker_action_df.columns:
        speaker_action_df["action_context"] = "All"

    grouped_by_bandit = speaker_action_df.groupby(["action_context"])

    for k, g in grouped_by_bandit:
        for s in speaker_list:
            speaker_col = "{}_prob".format(s.name)

            speaker_summary = {"speaker": s.name,
                               "action_context": k,
                               "prob_truthful": g[g.truthful][speaker_col].sum(),
                               "prob_optimal_action": (g.prob_optimal_action * g[speaker_col]).sum(),
                               "expected_rewards": (g.expected_rewards * g[speaker_col]).sum()}

            speaker_bandit_list.append(speaker_summary)

    return pd.DataFrame.from_records(speaker_bandit_list)
