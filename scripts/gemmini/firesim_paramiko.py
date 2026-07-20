#!/usr/bin/env python3
"""Paramiko helpers for FireSim remote access when system ssh is broken."""

from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path

import paramiko

DEFAULT_KEY = Path("/data/firesim_shared_key")
DEFAULT_HOST = "52.79.69.66"
DEFAULT_PORT = 443
DEFAULT_USER = "ubuntu"


def _load_key(key_path: Path = DEFAULT_KEY):
    errors = []
    for cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return cls.from_private_key_file(str(key_path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(f"Could not load SSH key {key_path}: {'; '.join(errors)}")


def connect(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    key_path: Path = DEFAULT_KEY,
) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=user, pkey=_load_key(key_path), timeout=30)
    return client


def run_remote(
    command: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    key_path: Path = DEFAULT_KEY,
) -> tuple[int, str, str]:
    client = connect(host=host, port=port, user=user, key_path=key_path)
    try:
        _, stdout, stderr = client.exec_command(f"bash -lc {shlex.quote(command)}", get_pty=False)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        return stdout.channel.recv_exit_status(), out, err
    finally:
        client.close()


def upload_file(
    local_path: Path,
    remote_path: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    key_path: Path = DEFAULT_KEY,
) -> None:
    local_path = local_path.resolve()
    client = connect(host=host, port=port, user=user, key_path=key_path)
    try:
        sftp = client.open_sftp()
        remote_parent = str(Path(remote_path).parent)
        parts = []
        current = remote_parent
        while current not in ("", "/"):
            parts.append(current)
            current = str(Path(current).parent)
        for directory in reversed(parts):
            try:
                sftp.stat(directory)
            except OSError:
                sftp.mkdir(directory)
        sftp.put(str(local_path), remote_path)
        if local_path.name.endswith(("-baremetal", "ort_test_firesim")) or os.access(local_path, os.X_OK):
            sftp.chmod(remote_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        sftp.close()
    finally:
        client.close()


def upload_tree(
    local_dir: Path,
    remote_dir: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    user: str = DEFAULT_USER,
    key_path: Path = DEFAULT_KEY,
) -> None:
    local_dir = local_dir.resolve()
    client = connect(host=host, port=port, user=user, key_path=key_path)
    try:
        sftp = client.open_sftp()

        def ensure_dir(path: str) -> None:
            if path in ("", "/"):
                return
            try:
                sftp.stat(path)
            except OSError:
                ensure_dir(str(Path(path).parent))
                sftp.mkdir(path)

        for root, _, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            target = remote_dir if rel == "." else f"{remote_dir.rstrip('/')}/{rel}"
            ensure_dir(target)
            for name in files:
                local_file = Path(root) / name
                remote_file = f"{target}/{name}"
                sftp.put(str(local_file), remote_file)
        sftp.close()
    finally:
        client.close()
