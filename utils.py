import ray
import time 

_LAST_FREE_TIME = 0.0
_TO_FREE = []

def ray_get_and_free(object_ids):
    """
    Call ray.get and then queue the object ids for deletion.

    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.

    Args:
        object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.

    Returns:
        The result of ray.get(object_ids).
    """

    free_delay_s = 10.0
    max_free_queue_size = 100

    global _LAST_FREE_TIME
    global _TO_FREE

    result = ray.get(object_ids)
    if type(object_ids) is not list:
        object_ids = [object_ids]
    _TO_FREE.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_TO_FREE) > max_free_queue_size
            or now - _LAST_FREE_TIME > free_delay_s):
        ray.internal.free(_TO_FREE)
        _TO_FREE = []
        _LAST_FREE_TIME = now

    return result

def get_vars(scope):
    import tensorflow as tf
    return [x for x in tf.global_variables() if scope in x.name]

def mlp(x, hidden_size, activation, out_activation=None):
    import tensorflow as tf
    for hidden in hidden_size[:-1]:
        x = tf.layers.dense(x, hidden, activation=activation)
    return tf.layers.dense(x, hidden_size[-1], activation=out_activation)

@ray.remote
class AsyncParameterServer():
    '''
    Asyncronous parameter server, 
    used for parameter sharing between learner and workers
    '''
    def __init__(self):
        self._param = None
        # hashing is a notation for paramter version
        self.hashing = int(time.time())
    
    def pull(self):
        return self._param
    
    def push(self, new_param):
        self._param = new_param
        self.hashing = int(time.time())
    
    def get_hashing(self):
        return self.hashing