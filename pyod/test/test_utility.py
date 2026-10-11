# -*- coding: utf-8 -*-


import os
import sys
import unittest

import numpy as np
# noinspection PyProtectedMember
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.metrics import precision_score
from sklearn.utils import check_random_state

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.utility import check_parameter
from pyod.utils.utility import standardizer
from pyod.utils.utility import get_label_n
from pyod.utils.utility import precision_n_scores
from pyod.utils.utility import argmaxn
from pyod.utils.utility import get_list_diff
from pyod.utils.utility import get_diff_elements
from pyod.utils.utility import get_intersection
from pyod.utils.utility import invert_order
from pyod.utils.utility import check_detector
from pyod.utils.utility import score_to_label


class TestParameters(unittest.TestCase):
    def setUp(self):
        pass

    def test_check_parameter_range(self):
        # verify parameter type correction
        with assert_raises(TypeError):
            check_parameter('f', 0, 100)

        with assert_raises(TypeError):
            check_parameter(1, 'f', 100)

        with assert_raises(TypeError):
            check_parameter(1, 0, 'f')

        with assert_raises(TypeError):
            check_parameter(argmaxn(value_list=[1, 2, 3], n=1), 0, 100)

        # if low and high are both unset
        with assert_raises(ValueError):
            check_parameter(50)

        # if low <= high
        with assert_raises(ValueError):
            check_parameter(50, 100, 99)

        with assert_raises(ValueError):
            check_parameter(50, 100, 100)

        # check one side
        with assert_raises(ValueError):
            check_parameter(50, low=100)
        with assert_raises(ValueError):
            check_parameter(50, high=0)

        assert_equal(True, check_parameter(50, low=10))
        assert_equal(True, check_parameter(50, high=100))

        # if check fails
        with assert_raises(ValueError):
            check_parameter(-1, 0, 100)

        with assert_raises(ValueError):
            check_parameter(101, 0, 100)

        with assert_raises(ValueError):
            check_parameter(0.5, 0.2, 0.3)

        # if check passes
        assert_equal(True, check_parameter(50, 0, 100))

        assert_equal(True, check_parameter(0.5, 0.1, 0.8))

        # if includes left or right bounds
        with assert_raises(ValueError):
            check_parameter(100, 0, 100, include_left=False,
                            include_right=False)
        assert_equal(True, check_parameter(0, 0, 100, include_left=True,
                                           include_right=False))
        assert_equal(True, check_parameter(0, 0, 100, include_left=True,
                                           include_right=True))
        assert_equal(True, check_parameter(100, 0, 100, include_left=False,
                                           include_right=True))
        assert_equal(True, check_parameter(100, 0, 100, include_left=True,
                                           include_right=True))

    def tearDown(self):
        pass


class TestScaler(unittest.TestCase):

    def setUp(self):
        random_state = check_random_state(42)
        self.X_train = random_state.rand(500, 5)
        self.X_test = random_state.rand(100, 5)
        self.X_test_diff = random_state.rand(100, 10)
        self.scores1 = [0.1, 0.3, 0.5, 0.7, 0.2, 0.1]
        self.scores2 = np.array([0.1, 0.3, 0.5, 0.7, 0.2, 0.1])

    def test_normalization(self):

        # test when X_t is presented and no scalar
        norm_X_train, norm_X_test = standardizer(self.X_train, self.X_test)
        assert_allclose(norm_X_train.mean(), 0, atol=0.05)
        assert_allclose(norm_X_train.std(), 1, atol=0.05)

        assert_allclose(norm_X_test.mean(), 0, atol=0.05)
        assert_allclose(norm_X_test.std(), 1, atol=0.05)

        # test when X_t is not presented and no scalar
        norm_X_train = standardizer(self.X_train)
        assert_allclose(norm_X_train.mean(), 0, atol=0.05)
        assert_allclose(norm_X_train.std(), 1, atol=0.05)

        # test when X_t is presented and the scalar is kept
        norm_X_train, norm_X_test, scalar = standardizer(self.X_train,
                                                         self.X_test,
                                                         keep_scalar=True)

        assert_allclose(norm_X_train.mean(), 0, atol=0.05)
        assert_allclose(norm_X_train.std(), 1, atol=0.05)

        assert_allclose(norm_X_test.mean(), 0, atol=0.05)
        assert_allclose(norm_X_test.std(), 1, atol=0.05)

        if not hasattr(scalar, 'fit') or not hasattr(scalar, 'transform'):
            raise AttributeError("%s is not a detector instance." % (scalar))

        # test when X_t is not presented and the scalar is kept
        norm_X_train, scalar = standardizer(self.X_train, keep_scalar=True)

        assert_allclose(norm_X_train.mean(), 0, atol=0.05)
        assert_allclose(norm_X_train.std(), 1, atol=0.05)

        if not hasattr(scalar, 'fit') or not hasattr(scalar, 'transform'):
            raise AttributeError("%s is not a detector instance." % (scalar))

        # test shape difference
        with assert_raises(ValueError):
            standardizer(self.X_train, self.X_test_diff)

    def test_invert_order(self):
        target = np.array([-0.1, -0.3, -0.5, -0.7, -0.2, -0.1]).ravel()
        scores1 = invert_order(self.scores1)
        assert_allclose(scores1, target)

        scores2 = invert_order(self.scores2)
        assert_allclose(scores2, target)

        target = np.array([0.6, 0.4, 0.2, 0, 0.5, 0.6]).ravel()
        scores2 = invert_order(self.scores2, method='subtraction')
        assert_allclose(scores2, target)

    def tearDown(self):
        pass


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        self.labels_ = [0.1, 0.2, 0.2, 0.8, 0.2, 0.5, 0.7, 0.9, 1, 0.3]
        self.labels_short_ = [0.1, 0.2, 0.2, 0.8, 0.2, 0.5, 0.7, 0.9, 1]
        self.manual_labels = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
        self.outliers_fraction = 0.3
        self.value_lists = [0.1, 0.3, 0.2, -2, 1.5, 0, 1, -1, -0.5, 11]

    def test_precision_n_scores(self):
        assert_equal(precision_score(self.y, self.manual_labels),
                     precision_n_scores(self.y, self.labels_))

    def test_get_label_n(self):
        assert_allclose(self.manual_labels,
                        get_label_n(self.y, self.labels_))

    def test_get_label_n_equal_3(self):
        manual_labels = [0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
        assert_allclose(manual_labels,
                        get_label_n(self.y, self.labels_, n=3))

    def test_inconsistent_length(self):
        with assert_raises(ValueError):
            get_label_n(self.y, self.labels_short_)

    def test_score_to_label(self):
        manual_scores = [0.1, 0.4, 0.2, 0.3, 0.5, 0.9, 0.7, 1, 0.8, 0.6]
        labels = score_to_label(manual_scores, outliers_fraction=0.1)
        assert_allclose(labels, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

        labels = score_to_label(manual_scores, outliers_fraction=0.3)
        assert_allclose(labels, [0, 0, 0, 0, 0, 1, 0, 1, 1, 0])

    def test_argmaxn(self):
        ind = argmaxn(self.value_lists, 3)
        assert_equal(len(ind), 3)

        ind = argmaxn(self.value_lists, 3)
        assert_equal(np.sum(ind), np.sum([4, 6, 9]))

        ind = argmaxn(self.value_lists, 3, order='asc')
        assert_equal(np.sum(ind), np.sum([3, 7, 8]))

        with assert_raises(ValueError):
            argmaxn(self.value_lists, -1)
        with assert_raises(ValueError):
            argmaxn(self.value_lists, 20)

    def test_get_list_diff(self):
        li1 = [1, 2, 3, 4]
        li2 = [2, 3, 4, 5]
        li3 = [8]

        ind = get_list_diff(li1, li2)
        assert (ind == [1])

        ind = get_list_diff(np.asarray(li2), np.asarray(li1))
        assert (ind == [5])

        ind = get_list_diff(li1, li1)
        assert (ind == [])

        ind = get_list_diff(li1, li3)
        assert (ind == [1, 2, 3, 4])

    def test_get_diff_elements(self):
        li1 = [1, 2, 3, 4]
        li2 = [2, 3, 4, 5]
        li3 = [8]

        ind = get_diff_elements(li1, li2)
        assert (ind == [1, 5])

        ind = get_diff_elements(np.asarray(li2), np.asarray(li1))
        assert (ind == [5, 1])

        ind = get_diff_elements(li1, li1)
        assert (ind == [])

        ind = get_diff_elements(li1, li3)
        assert (ind == [1, 2, 3, 4, 8])

    def test_get_intersection(self):
        li1 = [1, 2, 3, 4]
        li2 = [2, 3, 4, 5]
        li3 = [8]

        ind = get_intersection(li1, li2)
        assert (ind == [2, 3, 4])

        ind = get_intersection(np.asarray(li2), np.asarray(li1))
        assert (ind == [2, 3, 4])

        ind = get_intersection(li1, li1)
        assert (ind == [1, 2, 3, 4])

        ind = get_intersection(li1, li3)
        assert (ind == [])

    def tearDown(self):
        pass


class TestTiedScoreMetrics(unittest.TestCase):
    """Ties at the rank-n cutoff must not corrupt precision @ rank n
    (issue #760). The old percentile cut dropped the whole tied block, so
    the flagged count fell below n and precision divided by that smaller
    count."""

    def test_precision_n_scores_tie_at_cutoff(self):
        # one outlier on top, the other 99 scores identical: the tied block
        # fills the remaining 4 of n=5 slots, 4 of its 99 rows are outliers
        y = np.zeros(100)
        y[:5] = 1
        y_pred = np.array([1.0] + [0.0] * 99)
        assert_allclose(precision_n_scores(y, y_pred), 0.2323, atol=1e-4)

    def test_precision_n_scores_coarse_plateau(self):
        # 3 outliers on top, 7 outliers inside a 97-row plateau
        y = np.zeros(100)
        y[:10] = 1
        y_pred = np.array([1.0] * 3 + [0.0] * 97)
        assert_allclose(precision_n_scores(y, y_pred), 0.3505, atol=1e-4)

    def test_precision_n_scores_histogram_style(self):
        # 4 outliers at the top score, 20 rows tied at the cutoff score with
        # 6 outliers among them
        y = np.zeros(100)
        y_pred = np.zeros(100)
        y_pred[:4] = 3.0
        y[:4] = 1
        y_pred[4:24] = 2.0
        y[4:10] = 1
        assert_allclose(precision_n_scores(y, y_pred), 0.58, atol=1e-4)

    def test_precision_n_scores_all_tied_at_max(self):
        # 15 rows tied at the maximum hold all 10 outliers: every row of the
        # tied block is equally likely to sit in the top 10
        y = np.zeros(100)
        y_pred = np.zeros(100)
        y_pred[85:] = 1.0
        y[85:95] = 1
        assert_allclose(precision_n_scores(y, y_pred), 10 / 15, atol=1e-4)

    def test_precision_n_scores_constant_scores_report_prevalence(self):
        # a constant-score column carries no signal, so its expected
        # precision is the outlier prevalence, not 0.0
        y = np.zeros(100)
        y[:10] = 1
        y_pred = np.zeros(100)
        assert_allclose(precision_n_scores(y, y_pred), 0.1, atol=1e-4)

    def test_precision_n_scores_distinct_scores_unchanged(self):
        # with distinct scores the flagged rows are exactly the top n, so
        # the value must match the old behavior
        y = np.zeros(100)
        y_pred = np.linspace(0.05, 1.0, 100)
        y[[91, 93, 95, 97, 99]] = 1
        y[[3, 7, 11, 33, 55]] = 1
        assert_allclose(precision_n_scores(y, y_pred), 0.5)

        # perfect and inverted rankings are the two extremes
        y_perfect = np.zeros(100)
        y_perfect[:10] = 1
        assert_allclose(
            precision_n_scores(y_perfect, np.linspace(0, 1, 100)[::-1]), 1.0)

        y_inverted = np.zeros(100)
        y_inverted[:10] = 1
        assert_allclose(
            precision_n_scores(y_inverted, np.linspace(0, 1, 100)), 0.0)

    def test_precision_n_scores_explicit_n_with_ties(self):
        # n=3 with one outlier on top of a 9-row plateau that holds the
        # other two outliers
        y = np.zeros(10)
        y[:3] = 1
        y_pred = np.array([1.0] + [0.0] * 9)
        assert_allclose(precision_n_scores(y, y_pred, n=3), 13 / 27,
                        atol=1e-4)

    def test_precision_n_scores_n_out_of_range(self):
        y = [0, 1, 1, 0]
        y_pred = [0.1, 0.9, 0.8, 0.2]
        # n larger than the sample count was rejected before; keep that
        with assert_raises(ValueError):
            precision_n_scores(y, y_pred, n=10)
        with assert_raises(ValueError):
            precision_n_scores(y, y_pred, n=-2)
        # explicit n=0 is well-defined and stays 0.0
        assert_allclose(precision_n_scores(y, y_pred, n=0), 0.0)

    def test_get_label_n_flags_at_least_n(self):
        # the old percentile cut dropped the whole tied block and flagged
        # 1 row where n=5 was asked for
        y = np.zeros(100)
        y[:5] = 1
        y_pred = np.array([1.0] + [0.0] * 99)
        labels = get_label_n(y, y_pred)
        assert labels.sum() >= 5

        # all-tied scores: every row is at the cutoff, so no row can be
        # distinguished and the whole block is flagged
        y2 = np.zeros(100)
        y2[85:95] = 1
        y_pred2 = np.zeros(100)
        y_pred2[85:] = 1.0
        labels2 = get_label_n(y2, y_pred2)
        assert_equal(labels2.sum(), 15)

    def test_get_label_n_distinct_scores_unchanged(self):
        # the existing fixture behavior must survive the cutoff change
        y = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
        y_pred = [0.1, 0.2, 0.2, 0.8, 0.2, 0.5, 0.7, 0.9, 1, 0.3]
        manual_labels = [0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
        assert_allclose(get_label_n(y, y_pred), manual_labels)
        assert_allclose(get_label_n(y, y_pred, n=3),
                        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0])

    def test_score_to_label_tie_at_threshold(self):
        # 20 identical scores spanning the 10% cut: the old strict '>'
        # flagged 0 of 20 samples
        pred_scores = np.zeros(20)
        assert_equal(score_to_label(pred_scores, outliers_fraction=0.5),
                    np.ones(20))
        # a cut that lands exactly on a score now keeps that score
        pred_scores = np.array([0.0] * 9 + [5.0] + [9.0] * 10)
        # the interpolated 90th percentile of these 20 values is 9.0, a
        # score held by 10 rows: with >= those 10 rows are kept instead of
        # none, which is the top-half share the 10% cut asks for
        labels = score_to_label(pred_scores, outliers_fraction=0.1)
        assert_equal(labels.sum(), 10)

    def test_get_label_n_zero_outliers(self):
        # no outliers in y and n omitted: all-zero labels, no crash
        labels = get_label_n([0, 0, 0, 0], [0.1, 0.9, 0.8, 0.2])
        assert_equal(labels, np.zeros(4))

    def tearDown(self):
        pass


class TestCheckDetector(unittest.TestCase):

    def setUp(self):
        class DummyNegativeModel():
            def fit_negative(self):
                return

            def decision_function_negative(self):
                return

        class DummyPostiveModel():
            def fit(self):
                return

            def decision_function(self):
                return

        self.detector_positive = DummyPostiveModel()
        self.detector_negative = DummyNegativeModel()

    def test_check_detector_positive(self):
        check_detector(self.detector_positive)

    def test_check_detector_negative(self):
        with assert_raises(AttributeError):
            check_detector(self.detector_negative)


if __name__ == '__main__':
    unittest.main()
