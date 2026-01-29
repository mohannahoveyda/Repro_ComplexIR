import pickle, pathlib
from filelock import FileLock
import os, pickle, hashlib, pathlib, contextlib, fcntl


CACHE_DIR = pathlib.Path(os.environ.get("CACHE_DIR", ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class DiskKV:
    def __init__(self, filename: str):
        self.path = CACHE_DIR / filename
        self.lock = FileLock(str(self.path) + ".lock")
        if not self.path.exists():
            with self.lock:
                with self.path.open("wb") as fh:
                    pickle.dump({}, fh)

    def get(self, key, default=None):
        with self.lock:
            with self.path.open("rb") as fh:
                data = pickle.load(fh)
            return data.get(key, default)

    def put(self, key, value):
        with self.lock:
            with self.path.open("rb") as fh:
                data = pickle.load(fh)
            data[key] = value
            with self.path.open("wb") as fh:
                pickle.dump(data, fh)
def make_key(*parts) -> bytes:
    """
    Helper that turns an arbitrary list of *json-serialisable*
    pieces into a stable SHA1-bytes key.
    """
    blob = pickle.dumps(parts, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha1(blob).digest()
