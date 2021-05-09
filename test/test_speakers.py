import pandas as pd
import unittest

import simulations
from listeners import StatelessLiteralListener
from speakers import BeliefSpeaker, ActionSpeaker, CombinedSpeaker

test_colors = ['blue', 'green']
test_shapes = ['circle', 'square']
test_values = [-1, 1]
test_rewards = pd.Series({"green": 1, "blue": -1, "circle": 1, "square": -1})
test_features = test_colors + test_shapes
test_worlds = simulations.generate_worlds_from_feature_values(test_features, test_values)

literal_listener = StatelessLiteralListener(3, test_features, test_worlds)

action_tuples = [("green", "circle"), ("blue", "circle"), ("green", "square")]
action_context = simulations.generate_context_from_actions(action_tuples)

false_message = ("blue", 1)
true_message = ("green", 1)


class MyTestCase(unittest.TestCase):

    def test_belief_speaker_utility(self):

        test_speaker = BeliefSpeaker(literal_listener, beta=3, w=test_rewards, name="Belief")

        good_utility = test_speaker.utility(true_message, action_context)
        self.assertAlmostEqual(good_utility, -2.0794415, 3)

        bad_utility = test_speaker.utility(false_message, action_context)
        self.assertAlmostEqual(bad_utility, -23.0258509, 3)

    def test_action_speaker_utility(self):

        test_speaker = ActionSpeaker(literal_listener, beta=3, w=test_rewards, name="Action")

        good_utility = test_speaker.utility(true_message, action_context)
        self.assertAlmostEqual(good_utility, -0.7177, 3)

        bad_utility = test_speaker.utility(false_message, action_context)
        self.assertAlmostEqual(bad_utility, -3.0949, 3)

    def test_combined_speaker_utility(self):

        test_speaker = CombinedSpeaker(literal_listener, beta=3, w=test_rewards, name="Combined")

        good_utility = test_speaker.utility(true_message, action_context)
        self.assertAlmostEqual(good_utility, 0.97571, 3)

        bad_utility = test_speaker.utility(false_message, action_context)
        self.assertAlmostEqual(bad_utility, 0.090557, 3)


if __name__ == '__main__':
    unittest.main()
