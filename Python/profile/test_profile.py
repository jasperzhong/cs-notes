import cProfile
import io
import pstats

import numpy as np

pr = cProfile.Profile()
for _ in range(10):
    pr.enable()
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    z = x + y

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
