from typing import Any, List, Tuple

import pytest

from profile_extraction.util.utils import train_dev_test_split


class TestTrainTestValSplit:
    @pytest.mark.parametrize(
        "input,dev_split,test_split,expected",
        [
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.2, 0.2, (6, 2, 2)),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1, 0.2, (7, 1, 2)),
        ],
        ids=["60 to 20, 20 split", "60 to 10, 20 split"],
    )
    def test_should_split_correctly(
        self, input: List[Any], dev_split: float, test_split: float, expected: Tuple[int, int, int]
    ):
        len(input)
        train, dev, test = train_dev_test_split(input, dev_split=dev_split, test_split=test_split)
        print(train)
        print(dev)
        print(test)
        assert expected[0] == len(train)
        assert expected[1] == len(dev)
        assert expected[2] == len(test)

    def test_too_large_split_should_raise_error(self):
        dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        with pytest.raises(ValueError) as err:
            train_dev_test_split(dataset, 0.5, 0.5)
        assert str(err.value) == "The sum of the splits must be less than 1."

    def test_trainset_too_small_should_raise_error(self):
        dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        with pytest.raises(ValueError) as err:
            train_dev_test_split(dataset, 0.49, 0.5)
        assert str(err.value) == "The given split is not possible."
