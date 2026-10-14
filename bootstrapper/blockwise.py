import logging
import time
from multiprocessing import Event
from multiprocessing.pool import ThreadPool

import daisy
from daisy.cl_monitor import BlockFailure, CLMonitor
from daisy.tcp import IOLooper

logger = logging.getLogger(__name__)


def volara_log_dir(seg_dataset_prefix):
    """Volara log dir of one output prefix: <container>_volara_logs/<prefix in the container>
    beside the container, or a hashed name in the cwd when the path holds ':' or '='
    (daisy cannot pass those to workers)."""
    import hashlib

    container, _, group = seg_dataset_prefix.rpartition(".zarr")
    log_dir = f"{container}_volara_logs{group}" if container else f"{seg_dataset_prefix}_volara_logs"
    if ":" in log_dir or "=" in log_dir:
        log_dir = f"volara_logs_{hashlib.sha1(seg_dataset_prefix.encode()).hexdigest()[:8]}"
    return log_dir


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
        raise RuntimeError("; ".join(errors) + f"; worker logs in {daisy.logging.get_log_basedir()}")


class Monitor(CLMonitor):
    """daisy's progress bars under short task names, and one line per task in place of its
    Execution Summary. A failed or orphaned block keeps the full summary."""

    def __init__(self, server, names):
        self.names, self.start, self.seconds = names, {}, {}
        super().__init__(server)

    def on_task_start(self, task_id, task_state):
        self.start[task_id] = time.monotonic()
        super().on_task_start(self.names.get(task_id, task_id), task_state)

    def on_acquire_block(self, task_id, task_state):
        super().on_acquire_block(self.names.get(task_id, task_id), task_state)

    def on_release_block(self, task_id, task_state):
        super().on_release_block(self.names.get(task_id, task_id), task_state)

    def on_task_done(self, task_id, task_state):
        name = self.names.get(task_id, task_id)
        self.seconds[name] = time.monotonic() - self.start.get(task_id, time.monotonic())
        super().on_task_done(name, task_state)

    def on_block_failure(self, block, exception, context):
        name = self.names.get(block.block_id[0], block.block_id[0])
        self.summaries[name].block_failures.append(BlockFailure(block, exception, context["worker_id"]))

    def on_server_exit(self):
        if any(s.block_failures or s.state.failed_count or s.state.orphaned_count for s in self.summaries.values()):
            return super().on_server_exit()
        for progress in self.progresses.values():
            progress.close()
        for name, s in self.summaries.items():
            print(f"{name}: {s.state.completed_count} of {s.state.total_block_count} blocks done, "
                  f"{s.state.failed_count} failed, {self.seconds.get(name, 0):.1f} s")


def _serve(tasks, stop_event, names):
    server = daisy.Server(stop_event=stop_event)
    Monitor(server, names)
    return server.run_blockwise(tasks)


def run_blockwise(tasks, multiprocessing=True, name=None):
    """daisy.run_blockwise, but raise if any block failed or was orphaned. name, if given,
    stands for the task's id in what daisy prints."""
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
    names = {tasks[0].task_id: name} if name and len(tasks) == 1 else {}

    if multiprocessing:
        stop_event = Event()
        IOLooper.clear()
        with ThreadPool(processes=1) as pool:
            result = pool.apply_async(_serve, args=(tasks, stop_event, names))
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
        Monitor(server, names)
        task_states = server.run_blockwise(tasks)

    check_task_states(task_states)


def run_volara_task(task, multiprocessing=True, name=None):
    """volara BlockwiseTask.run_blockwise, but raise if any block failed.

    Always drop first: volara caches completed blocks on disk, so without
    this a re-run skips the stage and leaves stale (often empty) output even
    though the db was reset upstream.
    """
    task.drop()
    with task.task(multiprocessing=multiprocessing) as daisy_task:
        run_blockwise([daisy_task], multiprocessing=multiprocessing, name=name)
