import multiprocessing as mp
from multiprocessing.managers import BaseManager

from config import get_logger

log = get_logger('Launcher')

# ManagerServer 进程中的延迟初始化对象
_eval_queue = None
_result_queue = None
_cmd_queue = None
_task_queues = {}
_shm_pool_ref = None


def _get_eval_queue():
    global _eval_queue
    log.debug('get_eval_queue')
    if _eval_queue is None:
        _eval_queue = mp.Queue()
    return _eval_queue


def _get_result_queue():
    global _result_queue
    log.debug('get_result_queue')
    if _result_queue is None:
        _result_queue = mp.Queue()
    return _result_queue


def _get_command_queue():
    global _cmd_queue
    log.debug('get_command_queue')
    if _cmd_queue is None:
        _cmd_queue = mp.Queue()
    return _cmd_queue


def _get_task_queue(proj):
    global _task_queues
    log.debug('get_task_queue[%s]', proj)
    if proj not in _task_queues:
        _task_queues[proj] = mp.Queue()
    return _task_queues[proj]


def _get_shm_pool():
    global _shm_pool_ref
    log.debug('get_shm_pool')
    if _shm_pool_ref is None:
        from shared_memory_pool import SharedMemoryPool, SHM_NAMESPACE
        _shm_pool_ref = SharedMemoryPool(namespace=SHM_NAMESPACE)
    return _shm_pool_ref


class QueueManager(BaseManager):
    pass


QueueManager.register('get_eval_queue', callable=_get_eval_queue)
QueueManager.register('get_result_queue', callable=_get_result_queue)
QueueManager.register('get_command_queue', callable=_get_command_queue)
QueueManager.register('get_task_queue', callable=_get_task_queue)
QueueManager.register('get_shm_pool', callable=_get_shm_pool)
