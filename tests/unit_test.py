import pickle
import unittest
from nicety.conf import DotDict, dotdict_to_dict


# https://stackoverflow.com/a/31669594/10702372
def nested_equal(a, b):
    if not isinstance(b, type(a)):
        raise TypeError("The two input objects must be of same type")
    else:
        if isinstance(a, (list, tuple)):
            for aa, bb in zip(a, b):
                if not nested_equal(aa, bb):
                    return False
            return True
        else:
            raise NotImplementedError("Unsupported type: {}".format(type(a)))


class Tests(unittest.TestCase):
    def test_dot_dict(self):
        # nested
        original = {"a": {"b": [1, 2, {"c": 3}]}}
        d = DotDict(original)

        assert d.a.b[0] == 1
        assert d.a.b[2].c == 3

        dump = pickle.dumps(d)
        restored = pickle.loads(dump)

        assert d == restored

        # this should not raise an error
        getattr(d, "welp", None)
