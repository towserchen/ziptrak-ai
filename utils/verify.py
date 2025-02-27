import hmac
import hashlib
import time

def generate_access_token(secret, timestamp):
    return hmac.new(secret.encode(), str(timestamp).encode(), hashlib.sha512).hexdigest()


def verify_access_token(secret, received_token, timestamp):
    time_stamps = [timestamp - 1, timestamp, timestamp + 1]
    
    valid_tokens = [hmac.new(secret.encode(), str(t).encode(), hashlib.sha512).hexdigest() for t in time_stamps]

    return received_token in valid_tokens