def fib(n):
    """Return the n-th Fibonacci number (0-indexed).

    >>> fib(0)
    0
    >>> fib(1)
    1
    >>> fib(6)
    8
    """
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a