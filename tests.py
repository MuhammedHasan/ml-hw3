import unittest
import main_script
from sklearn.metrics import accuracy_score


class TestMainScript(unittest.TestCase):

    def test_a(self):
        exp_missing_features = [0, 35, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 3, 3, 0, 2, 2, 0, 0, 4]

        self.assertEqual(exp_missing_features, main_script.missing_features)

    def test_b_c_d(self):
        main_script.df.to_csv('df.csv')
        main_script.df_b.to_csv('df_b.csv')
        main_script.df_c.to_csv('df_c.csv')
        main_script.df_d.to_csv('df_d.csv')

    def test_e(self):
        self.assertEqual(1, main_script.training_accuracy)
        self.assertAlmostEqual(0.72, main_script.test_accuracy)
        print
        print 'train accuracy_score', main_script.training_accuracy
        print 'test accuracy_score', main_script.test_accuracy

    def test_f(self):
        exp_len = len(main_script.df_d.columns) - 1
        self.assertEqual(exp_len, len(main_script.feature_importance))
        # print sorted(main_script.feature_importance,
        #              key=lambda x: x[1], reverse=True)

    def test_g(self):
        main_script.visualise_line_graph()
        self.assertEqual(61, main_script.num_of_features)

    def test_h_its_own(self):
        model = main_script.train_my_model()
        training_accuracy = accuracy_score(
            model.predict(main_script.X_train), main_script.y_train)
        test_accuracy = accuracy_score(
            model.predict(main_script.X_test), main_script.y_test)
        print
        print 'custom train:', training_accuracy
        print 'custom test:', test_accuracy

    def test_h(self):
        for _ in range(5):
            predicted_train_y = main_script.custom_model_prediction(
                'dataset_test.csv')
            real_y = map(int, open('dataset_test_y.txt').read().split(','))

            print list(predicted_train_y)
            print list(real_y)
            print accuracy_score(predicted_train_y, real_y)

    def test_sample_test_set(self):
        predicted_train_y = main_script.custom_model_prediction(
            'sample_test.txt')
        self.assertIsNotNone(predicted_train_y)


if __name__ == '__main__':
    unittest.main()
