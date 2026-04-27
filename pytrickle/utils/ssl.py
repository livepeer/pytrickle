"""SSL/TLS utilities for pytrickle servers.

Supports self-signed certificate generation (via openssl CLI or cryptography
library) and loading user-provided certificate files.
"""

import datetime
import ipaddress
import logging
import os
import shutil
import ssl
import subprocess
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def setup_ssl_context(
    enabled: bool = False,
    certfile: Optional[str] = None,
    keyfile: Optional[str] = None,
) -> Optional[ssl.SSLContext]:
    """Create and return an SSL context for the server.

    If *certfile* and *keyfile* are provided, loads them.
    If *enabled* is ``True`` but no files are provided, generates a
    self-signed certificate. Returns ``None`` if SSL is disabled.

    Args:
        enabled: Whether to enable HTTPS.
        certfile: Path to SSL certificate file (PEM format).
        keyfile: Path to SSL private key file (PEM format).

    Returns:
        A configured :class:`ssl.SSLContext` or ``None``.
    """
    if not enabled:
        return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    if certfile and keyfile:
        if not os.path.isfile(certfile):
            raise FileNotFoundError(f"SSL certificate file not found: {certfile}")
        if not os.path.isfile(keyfile):
            raise FileNotFoundError(f"SSL key file not found: {keyfile}")
        ctx.load_cert_chain(certfile, keyfile)
        logger.info("Loaded SSL certificate from %s", certfile)
    else:
        cert_path, key_path = generate_self_signed_cert()
        cert_dir = os.path.dirname(cert_path)
        key_dir = os.path.dirname(key_path)
        try:
            ctx.load_cert_chain(cert_path, key_path)
            logger.info("Using auto-generated self-signed SSL certificate")
        finally:
            # Clean up temporary cert files and the temporary directory
            for path in (cert_path, key_path):
                try:
                    os.remove(path)
                except OSError:
                    pass

            if cert_dir and cert_dir == key_dir:
                try:
                    os.rmdir(cert_dir)
                except OSError:
                    pass
    return ctx


def generate_self_signed_cert() -> Tuple[str, str]:
    """Generate a temporary self-signed certificate and key.

    Tries the ``openssl`` CLI first, then falls back to the *cryptography*
    library.

    Returns:
        Tuple of ``(cert_file_path, key_file_path)``.
    """
    if shutil.which("openssl"):
        return _generate_self_signed_cert_openssl()

    # Fallback to cryptography library if available
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        raise RuntimeError(
            "SSL is enabled but neither 'openssl' CLI nor 'cryptography' "
            "library is available. Install cryptography "
            "(pip install cryptography) or ensure openssl is on PATH."
        )

    return _generate_self_signed_cert_cryptography()


def _generate_self_signed_cert_openssl() -> Tuple[str, str]:
    """Generate a self-signed cert using the ``openssl`` CLI."""
    tmpdir = tempfile.mkdtemp(prefix="pytrickle_ssl_")
    cert_path = os.path.join(tmpdir, "cert.pem")
    key_path = os.path.join(tmpdir, "key.pem")

    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", key_path, "-out", cert_path,
        "-days", "1", "-nodes", "-subj", "/CN=localhost",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to generate self-signed certificate: {exc.stderr}"
        ) from exc

    return cert_path, key_path


def _generate_self_signed_cert_cryptography() -> Tuple[str, str]:
    """Generate a self-signed cert using the *cryptography* library."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv6Address("::1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    tmpdir = tempfile.mkdtemp(prefix="pytrickle_ssl_")
    cert_path = os.path.join(tmpdir, "cert.pem")
    key_path = os.path.join(tmpdir, "key.pem")

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    return cert_path, key_path
