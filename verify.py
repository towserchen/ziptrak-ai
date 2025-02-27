import hmac
import hashlib
import time

def generate_access_token(secret, timestamp):
    return hmac.new(secret.encode(), str(timestamp).encode(), hashlib.sha512).hexdigest()


def verify_access_token(secret, received_token, timestamp):
    time_stamps = [timestamp - 1, timestamp, timestamp + 1]
    
    valid_tokens = [hmac.new(secret.encode(), str(t).encode(), hashlib.sha512).hexdigest() for t in time_stamps]

    return received_token in valid_tokens

# 示例：生成和验证 access_tokenW
secret = 'mySecretKey'
timestamp = int(time.time())  # 获取当前秒的时间戳

# 生成 token
access_token = generate_access_token(secret, timestamp)
print(f"Generated Access Token: {access_token}")

# 验证 token
is_valid = verify_access_token(secret, access_token, timestamp)
print(f"Is the token valid? {'Yes' if is_valid else 'No'}")
