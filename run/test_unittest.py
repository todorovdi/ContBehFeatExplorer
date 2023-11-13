import unittest

def func_name(a1, a2):
    return a1+a2

class test1 ( unittest.TestCase ):

    # before every single test
    def setUp(self):
        pass

    def tearDown(self):
        pass


    # should be started with "test_"
    def test_smth(self):
        res = func_name()
        self.assertEqual(res,0)

        arg1 = 2; arg = 44

        # test exceptions
        # we exc types that I know exsit in the function
        self.assertRaises(ValueError, func_name, arg1, arg2)

        # OR
        with assertRaises(ValueError):
            func_name(arg1, arg2)

        a = (
            "fdfd"
            "fds"
        )

if __name__ == '__main__':
    unittest.main()

# python -m unittest test_test.py
# or just python test_test.py
