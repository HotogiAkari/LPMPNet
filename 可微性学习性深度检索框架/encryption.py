import numpy as np
import cv2
import os
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.PublicKey import RSA
import random

# ================
# AES 加密/解密
# ================
def aes_encrypt(img: np.ndarray, key=None, mode="CBC"):
    """
    使用 AES 加密整张图像
    mode: "CBC" or "CTR"
    """
    if key is None:
        key = get_random_bytes(32)  # AES-256
    
    h, w, c = img.shape
    flat = img.flatten().tobytes()

    # padding (PKCS7)
    pad_len = 16 - (len(flat) % 16)
    flat_padded = flat + bytes([pad_len]) * pad_len

    iv = get_random_bytes(16)

    if mode == "CBC":
        cipher = AES.new(key, AES.MODE_CBC, iv)
    elif mode == "CTR":
        cipher = AES.new(key, AES.MODE_CTR)
    else:
        raise ValueError("Unsupported AES mode")

    enc_bytes = cipher.encrypt(flat_padded)
    enc_arr = np.frombuffer(enc_bytes[:len(flat)], dtype=np.uint8).reshape(h, w, c)

    return enc_arr, key, iv


def aes_decrypt(enc_img: np.ndarray, key, iv, mode="CBC"):
    """
    AES 解密
    """
    h, w, c = enc_img.shape
    enc_bytes = enc_img.flatten().tobytes()

    if mode == "CBC":
        cipher = AES.new(key, AES.MODE_CBC, iv)
    elif mode == "CTR":
        cipher = AES.new(key, AES.MODE_CTR, nonce=iv[:8])
    else:
        raise ValueError("Unsupported AES mode")

    dec_padded = cipher.decrypt(enc_bytes)

    # remove PKCS7 padding
    pad_len = dec_padded[-1]
    dec_bytes = dec_padded[:-pad_len]

    dec_arr = np.frombuffer(dec_bytes, dtype=np.uint8).reshape(h, w, c)
    return dec_arr


# ================
# RSA + AES 混合加密
# ================
def generate_rsa_keys(key_size=2048):
    """
    生成 RSA 公私钥
    """
    key = RSA.generate(key_size)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return public_key, private_key


def hybrid_encrypt(img: np.ndarray, public_key, mode="CBC"):
    """
    混合加密：RSA 加密 AES key，AES 加密图像
    """
    # 生成 AES key
    aes_key = get_random_bytes(32)
    iv = get_random_bytes(16)

    # 加密图像
    enc_img, _, _ = aes_encrypt(img, key=aes_key, mode=mode)

    # RSA 加密 AES key
    rsa_key = RSA.import_key(public_key)
    cipher_rsa = PKCS1_OAEP.new(rsa_key)
    enc_key = cipher_rsa.encrypt(aes_key)

    return enc_img, enc_key, iv


def hybrid_decrypt(enc_img, enc_key, iv, private_key, mode="CBC"):
    """
    混合解密
    """
    rsa_key = RSA.import_key(private_key)
    cipher_rsa = PKCS1_OAEP.new(rsa_key)
    aes_key = cipher_rsa.decrypt(enc_key)

    dec_img = aes_decrypt(enc_img, aes_key, iv, mode=mode)
    return dec_img


# ================
# 置乱操作
# ================
def block_permutation(img: np.ndarray, block_size=16):
    h, w, c = img.shape
    h_blocks, w_blocks = h // block_size, w // block_size
    blocks = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, :]
            blocks.append(block)
    random.shuffle(blocks)
    new_img = np.zeros_like(img)
    idx = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            new_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size, :] = blocks[idx]
            idx += 1
    return new_img


def intra_block_permutation(img: np.ndarray, block_size=16):
    h, w, c = img.shape
    new_img = np.zeros_like(img)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size, :]
            flat = block.reshape(-1, c)
            np.random.shuffle(flat)
            new_img[i:i+block_size, j:j+block_size, :] = flat.reshape(block.shape)
    return new_img


def global_pixel_permutation(img: np.ndarray):
    h, w, c = img.shape
    flat = img.reshape(-1, c)
    np.random.shuffle(flat)
    return flat.reshape(h, w, c)
