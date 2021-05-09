import unittest
import simulations

test_colors = ['blue', 'green']
test_shapes = ['circle', 'square']


class MyTestCase(unittest.TestCase):

    def test_actions_from_features(self):

        test_actions = simulations.generate_actions_from_features(test_colors, test_shapes)
        correct_test_actions = [('blue', 'circle'), ('blue', 'square'), ('green', 'circle'), ('green', 'square')]

        self.assertListEqual(test_actions, correct_test_actions)

    def test_context_from_actions(self):

        action_tuples = [("green", "circle"), ("blue", "circle"), ("green", "square")]
        action_context = simulations.generate_context_from_actions(action_tuples)

        action_names = ['green circle', 'blue circle', 'green square']

        # Assert that we get the correct list of action names back out
        self.assertListEqual(list(action_context.index.values), action_names)

        # Assert that we get the right features for each action
        blue_circle = action_context.loc['blue circle']
        self.assertEqual(blue_circle['circle'], 1)
        self.assertEqual(blue_circle['blue'], 1)
        self.assertEqual(blue_circle['green'], 0)
        self.assertEqual(blue_circle['square'], 0)

    def test_worlds_from_features(self):

        test_features = test_colors + test_shapes
        test_values = [-1, 1]

        test_worlds = simulations.generate_worlds_from_feature_values(test_features, test_values)

        # We should get num_features ^ num_values worlds
        correct_num_worlds = len(test_features)**len(test_values)
        self.assertEqual(len(test_worlds), correct_num_worlds)


if __name__ == '__main__':
    unittest.main()
