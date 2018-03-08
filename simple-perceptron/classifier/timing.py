from timeit import Timer


"""
https://stackoverflow.com/questions/20974147/timeit-eats-return-value
"""


def time_method(func, *args):
    output_container = []

    def wrapper():
        output_container.append(func(*args))

    timer = Timer(wrapper)
    delta = timer.timeit(number=30)

    return delta, output_container.pop()
