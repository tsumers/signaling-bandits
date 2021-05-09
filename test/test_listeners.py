import pandas as pd
import unittest

import simulations
from listeners import StatelessLiteralListener

test_colors = ['blue', 'green']
test_shapes = ['circle', 'square']

test_action_context = simulations.generate_context_from_actions([("green", "circle"),
                                                                 ("blue", "circle"),
                                                                 ("green", "square")])


class MyTestCase(unittest.TestCase):

    def test_literal_listener_conditioning(self):

        test_features = test_colors + test_shapes
        test_values = [-1, 1]

        test_worlds = simulations.generate_worlds_from_feature_values(test_features, test_values)
        test_green = ('green', 1)

        conditioned_worlds = StatelessLiteralListener.condition_worlds_on_message(test_worlds, test_green)

        # Conditioning on one value should halve the number of possible worlds
        original_num_worlds = len(test_features)**len(test_values)
        conditioned_num_worlds = original_num_worlds / 2
        self.assertEqual(len(conditioned_worlds), conditioned_num_worlds)

        # Conditioning on another value should halve it again
        test_blue = ('blue', -1)
        reconditioned_worlds = StatelessLiteralListener.condition_worlds_on_message(conditioned_worlds, test_blue)
        self.assertEqual(len(reconditioned_worlds), conditioned_num_worlds / 2)

    def test_listener_reward_estimation(self):

        rewards = pd.Series({"blue": -1, "green": 1, "circle": 1, "square": -1})
        estimated_values = StatelessLiteralListener.estimate_rewards(rewards, test_action_context)

        self.assertEqual(list(estimated_values), [2, 0, 0])

    def test_conditioned_actions(self):

        test_worlds = simulations.generate_worlds_from_feature_values(test_colors + test_shapes, [-1, 1])
        literal_listener = StatelessLiteralListener(3, test_colors + test_shapes, test_worlds)

        test_message = ('green', 1)
        green_positive = literal_listener.action_policy(test_message, test_action_context)
        results = [0.4879, 0.0243, 0.4879]

        for result in zip(results, green_positive.listener_choice_prob.values):
            self.assertAlmostEqual(result[0], result[1], 3)

        test_message = ('blue', 1)
        blue_positive = literal_listener.action_policy(test_message, test_action_context)
        results = [0.0453, 0.9094, 0.0453]

        for result in zip(results, blue_positive.listener_choice_prob.values):
            self.assertAlmostEqual(result[0], result[1], 3)


if __name__ == '__main__':
    unittest.main()
