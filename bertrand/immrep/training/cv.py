from bertrand.immrep.sample_test_set import sample_test, sample_test_additional
from bertrand.immrep.sample_train_set import sample_train, sample_train_additional


def train_test_generator(train, test, NTEST=7, NTRAIN=1):
    for test_iteration in range(NTEST):
        test_sample = sample_test(train, test, seed=test_iteration)
        print(f"Test sample {test_iteration}")
        for train_iteration in range(NTRAIN):
            print(f"Train sample {train_iteration}")
            train_sample = sample_train(train, test, test_sample)
            yield dict(
                train_sample=train_sample,
                test_sample=test_sample,
                test_iteration=test_iteration,
                train_iteration=train_iteration,
            )


def train_test_additional_generator(train, test, NTEST=7, NTRAIN=1):
    for test_iteration in range(NTEST):
        test_sample = sample_test_additional(train, test, seed=test_iteration)
        print(f"Test sample {test_iteration}")
        for train_iteration in range(NTRAIN):
            print(f"Train sample {train_iteration}")
            train_sample = sample_train_additional(train, test, test_sample)
            yield dict(
                train_sample=train_sample,
                test_sample=test_sample,
                test_iteration=test_iteration,
                train_iteration=train_iteration,
            )