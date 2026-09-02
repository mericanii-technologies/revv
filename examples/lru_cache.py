import threading
import time
from collections import OrderedDict


class LRUCache:
    """
    A thread-safe LRU cache with per-entry TTL expiry.
    """

    def __init__(self, maxsize, default_ttl):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._cache = OrderedDict()  # key -> (value, expiry_time)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _is_expired(self, expiry_time):
        return time.time() >= expiry_time

    def get(self, key):
        with self._lock:
            if key in self._cache:
                value, expiry_time = self._cache[key]
                if self._is_expired(expiry_time):
                    # Entry expired
                    del self._cache[key]
                    self._expirations += 1
                    self._misses += 1
                    return None
                else:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
            else:
                self._misses += 1
                return None

    def put(self, key, value, ttl=None):
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            expiry_time = time.time() + ttl

            if key in self._cache:
                # Update existing entry
                self._cache[key] = (value, expiry_time)
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = (value, expiry_time)
                # Evict if over capacity
                while len(self._cache) > self.maxsize:
                    self._cache.popitem(last=False)
                    self._evictions += 1

    def stats(self):
        with self._lock:
            return {
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'expirations': self._expirations
            }


import unittest


class TestLRUCache(unittest.TestCase):

    def test_lru_eviction_order(self):
        cache = LRUCache(maxsize=2, default_ttl=10)
        cache.put('a', 1)
        cache.put('b', 2)
        # Access 'a' to make it most recently used
        cache.get('a')
        # Add 'c', should evict 'b' (least recently used)
        cache.put('c', 3)
        self.assertEqual(cache.get('a'), 1)
        self.assertIsNone(cache.get('b'))
        self.assertEqual(cache.get('c'), 3)

    def test_ttl_expiry(self):
        cache = LRUCache(maxsize=10, default_ttl=10)
        cache.put('key', 'value', ttl=0.1)
        self.assertEqual(cache.get('key'), 'value')
        time.sleep(0.15)
        self.assertIsNone(cache.get('key'))

    def test_thread_safety(self):
        cache = LRUCache(maxsize=100, default_ttl=10)
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    key = f"key_{thread_id}_{i}"
                    cache.put(key, i)
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

    def test_stats_accuracy(self):
        cache = LRUCache(maxsize=2, default_ttl=10)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.get('a')  # hit
        cache.get('c')  # miss
        cache.put('d', 4)  # evicts 'b'
        stats = cache.stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['evictions'], 1)
        self.assertEqual(stats['expirations'], 0)

    def test_edge_cases(self):
        # Test with maxsize=1
        cache = LRUCache(maxsize=1, default_ttl=10)
        cache.put('a', 1)
        cache.put('b', 2)
        self.assertIsNone(cache.get('a'))
        self.assertEqual(cache.get('b'), 2)

        # Test updating existing key
        cache.put('b', 3)
        self.assertEqual(cache.get('b'), 3)

        # Test None value
        cache.put('c', None)
        self.assertIsNone(cache.get('c'))


if __name__ == '__main__':
    unittest.main()
