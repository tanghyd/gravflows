# https://stackoverflow.com/questions/63762655/how-to-create-a-continuous-stream-of-pythons-concurrent-futures-processpoolexec
import concurrent.futures as cf
import threading
import time
from itertools import count

import numpy as np
from numpy.random import SeedSequence, default_rng


def dojob(process, iterations, samples, rg):
    # Do some tasks
    result = []
    for i in range(iterations):
        a = rg.standard_normal(samples)
        b = rg.integers(-3, 3, samples)
        mean = np.mean(a + b)
        result.append((i, mean))
    return {process: result}


def execute_concurrently(cpus, max_queue_length, get_job_fn, process_result_fn):
    running_futures = set()
    jobs_complete = 0
    job_cond = threading.Condition()
    all_complete_event = threading.Event()

    def on_complete(future):
        nonlocal jobs_complete
        if process_result_fn(future.result()):
            all_complete_event.set()
        running_futures.discard(future)
        jobs_complete += 1
        with job_cond:
            job_cond.notify_all()

    time_since_last_status = 0
    start_time = time.time()
    with cf.ProcessPoolExecutor(cpus) as executor:
        while True:
            while len(running_futures) < max_queue_length:
                fn, args = get_job_fn()
                fut = executor.submit(fn, *args)
                fut.add_done_callback(on_complete)
                running_futures.add(fut)

            # with job_cond:
            #     job_cond.wait()

            try:
                with job_cond:
                    job_cond.wait()
            except KeyboardInterrupt:
                # Cancel running futures
                for future in running_futures:
                    _ = future.cancel()
                # Ensure concurrent.futures.executor jobs really do finish.
                _ = cf.wait(running_futures, timeout=None)
                
            if all_complete_event.is_set():
                break

            if time.time() - time_since_last_status > 1.0:
                rps = jobs_complete / (time.time() - start_time)
                print(
                    f"{len(running_futures)} running futures on {cpus} CPUs, "
                    f"{jobs_complete} complete. RPS: {rps:.2f}"
                )
                time_since_last_status = time.time()


def main():
    ss = SeedSequence(1234567890)
    counter = count(start=0, step=1)
    iterations = 10000
    samples = 1000
    results = []

    def get_job():
        seed = ss.spawn(1)[0]
        rg = default_rng(seed)
        process = next(counter)
        return dojob, (process, iterations, samples, rg)

    def process_result(result):
        for k, v in result.items():
            results.append(np.std(v)) 
        if len(results) >= 10000:
            return True  # signal we're complete

    execute_concurrently(
        cpus=4,
        max_queue_length=20,
        get_job_fn=get_job,
        process_result_fn=process_result,
    )


if __name__ == "__main__":
    main()