from tqdm import tqdm
from joblib import Parallel, delayed


def parallelize(generator, total, n_jobs=-1):
    return list(
        tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(generator),
            total=total,
            leave=False,
        )
    )


if __name__ == "__main__":
    parallelize((delayed(print)(i) for i in range(10)), 10, n_jobs=2)
