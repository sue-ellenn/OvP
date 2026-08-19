import hashlib


def content_hash(html): return hashlib.sha256(html.encode("utf-8")).hexdigest()
