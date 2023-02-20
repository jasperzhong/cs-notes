## Pybind11 Rlease GIL

```sh
$ python test.py
Result for 30: 5.7646075129679066e+17 (thread 0)
Result for 30: 5.7646075129679066e+17 (thread 1)
Result for 30: 5.7646075129679066e+17 (thread 2)
Result for 30: 5.7646075129679066e+17 (thread 3)
Result for 30: 5.7646075129679066e+17 (thread 4)
Testing with release_gil=False took 18.07945489883423 seconds
Result for 30: 5.7646075129679066e+17 (thread 2)
Result for 30: 5.7646075129679066e+17 (thread 0)
Result for 30: 5.7646075129679066e+17 (thread 3)
Result for 30: 5.7646075129679066e+17 (thread 4)
Result for 30: 5.7646075129679066e+17 (thread 1)
Testing with release_gil=True took 3.6139564514160156 seconds
```

The experiment shows that C++ function invoked from Python are running in parallel when GIL is released.
