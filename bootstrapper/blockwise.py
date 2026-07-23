import logging
from multiprocessing import Event
from multiprocessing.pool import ThreadPool

import daisy
from daisy.cl_monitor import CLMonitor
from daisy.tcp import IOLooper

logger = logging.getLogger(__name__)


def check_task_states(task_states):
    # daisy 1.x counts failed and orphaned blocks as done, so its
    # run_blockwise bool is True even when every block failed
    errors = [
        f"task {task_id}: {ts.failed_count} failed, {ts.orphaned_count} orphaned "
        f"of {ts.total_block_count} blocks"
        for task_id, ts in task_states.items()
        if ts.failed_count > 0 or ts.orphaned_count > 0
    ]
    if errors:
        raise RuntimeError("; ".join(errors))


def _serve(tasks, stop_event):
    server = daisy.Server(stop_event=stop_event)
    CLMonitor(server)
    return server.run_blockwise(tasks)


def run_blockwise(tasks, multiprocessing=True):
    """daisy.run_blockwise, but raise if any block failed or was orphaned."""
    # expand and dedup upstream tasks, like daisy.run_blockwise
    task_ids = set()
    all_tasks = []
    while len(tasks) > 0:
        task, tasks = tasks[0], tasks[1:]
        if task.task_id not in task_ids:
            task_ids.add(task.task_id)
            all_tasks.append(task)
        tasks.extend(task.upstream_tasks)
    tasks = all_tasks

    if multiprocessing:
        stop_event = Event()
        IOLooper.clear()
        with ThreadPool(processes=1) as pool:
            result = pool.apply_async(_serve, args=(tasks, stop_event))
            try:
                task_states = result.get()
            except KeyboardInterrupt:
                # stop gracefully, then propagate so an interrupted run
                # cannot exit 0 (interrupted blocks are pending, not failed)
                stop_event.set()
                check_task_states(result.get())
                raise
    else:
        server = daisy.SerialServer()
        CLMonitor(server)
        task_states = server.run_blockwise(tasks)

    check_task_states(task_states)


def run_volara_task(task, multiprocessing=True):
    """volara BlockwiseTask.run_blockwise, but raise if any block failed.

    Always drop first: volara caches completed blocks on disk, so without
    this a re-run skips the stage and leaves stale (often empty) output even
    though the db was reset upstream.
    """
    task.drop()
    with task.task(multiprocessing=multiprocessing) as daisy_task:
        run_blockwise([daisy_task], multiprocessing=multiprocessing)
