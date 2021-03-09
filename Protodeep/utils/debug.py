
import time
import functools
import atexit


timers = []


# def timer(func):
#     """Print the runtime of the decorated function"""
#     @functools.wraps(func)
#     def wrapper_timer(*args, **kwargs):
#         start_time = time.perf_counter()    # 1
#         value = func(*args, **kwargs)
#         end_time = time.perf_counter()      # 2
#         run_time = end_time - start_time    # 3
#         print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
#         return value
#     return wrapper_timer

def class_timer(obj):
    if isinstance(obj, type) is False:
        # print(obj)
        return timer(obj)
    for name, method in obj.__dict__.items():
        # print(name, method)
        if callable(method):
            setattr(obj, name, timer(getattr(obj, name), obj.__name__))
    return obj


def print_timers_logs():
    for t in timers:
        t.print_logs()


atexit.register(print_timers_logs)


class Timer:

    run_time = 0
    call_count = 0

    def __init__(self, func, class_name, *args, **kwargs):
        self.func = func
        self.class_name = class_name
        timers.append(self)

    def print_logs(self):
        if self.call_count > 0:
            avg_time = self.run_time / self.call_count
            full_name = ''
            if self.class_name is not None:
                full_name += self.class_name + '.'
            full_name += self.func.__name__
            print(
                f"{full_name.ljust(30)}: "
                f"total run time {self.run_time:.6f} s - "
                f"Called {str(self.call_count).ljust(8)} times - "
                f"Average time {avg_time:.6f} s"
            )


def timer(func, class_name=None):

    T = Timer(func, class_name)

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        T.call_count += 1
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        T.run_time += end_time - start_time
        return value
    return wrapper_timer


# class Debug:

#     # def __init__(self, func):
#     #     functools.update_wrapper(self, func)
#     #     self.func = func
#     #     self.num_calls = 0
#     times = dict()

#     @classmethod
#     def timer(mode='total'):
#         def timer_decorator(func):
#             """Print the runtime of the decorated function"""
#             @functools.wraps(func)
#             def wrapper_timer(*args, **kwargs):
#                 start_time = time.perf_counter()    # 1
#                 value = func(*args, **kwargs)
#                 end_time = time.perf_counter()      # 2
#                 run_time = end_time - start_time    # 3
#                 print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
#                 return value

#             def wrapper_timer_total(*args, **kwargs):
#                 start_time = time.perf_counter()    # 1
#                 value = func(*args, **kwargs)
#                 end_time = time.perf_counter()      # 2
#                 run_time = end_time - start_time    # 3
#                 self.times[func.__name__] += run_time
#                 # print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
#                 return value

#             if mode == 'total':
#                 return wrapper_timer_total
#             else:
#                 return wrapper_timer
#             return wrapper_timer

#     def __call__(self, *args, **kwargs):
#         self.num_calls += 1
#         print(f"Call {self.num_calls} of {self.func.__name__!r}")
#         print(args, kwargs)
#         start_time = time.perf_counter()
#         value = self.func(*args)
#         end_time = time.perf_counter()
#         run_time = end_time - start_time
#         print(f"Finished {self.func.__name__!r} in {run_time:.4f} secs")
#         return value
#         # return self.func(*args, **kwargs)
#         # quit()

#         # self.timer(func)

#     # def __call__
#     # def __call__(self, func):
#     #     @functools.wraps(func)
#     #     def wrapper_timer(*args, **kwargs):
#     #         start_time = time.perf_counter()
#     #         value = func(*args, **kwargs)
#     #         end_time = time.perf_counter()
#     #         run_time = end_time - start_time
#     #     return wrapper_timer
#     #     return self.timer(func)

#     # def timer(self, func):
