import threading
import time

import example


def test_function_wrapper(x, thread, release_gil=True):
    if release_gil:
        result = example.test_function_release_gil(x)
    else:
        result = example.test_function(x)
    print(f"Result for {x}: {result} (thread {thread})")

def test(release_gil):
    start = time.time()
    threads = []
    for i in range(5):
        t = threading.Thread(target=test_function_wrapper, args=(30, i, release_gil))
        threads.append(t)
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    end = time.time()
    print(f"Testing with release_gil={release_gil} took {end - start} seconds")

if __name__ == "__main__":
    test(release_gil=False)
    test(release_gil=True)
