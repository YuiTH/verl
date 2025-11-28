import base64
import logging

import msal
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

import base64, hashlib
from typing import Optional, Dict
from azure.keyvault.secrets import SecretClient
from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

def build_msal_cert_credentials_from_kv(
    keyvault_url: str,
    cert_name: str,
    credential: Optional[TokenCredential] = None,
    pfx_password: Optional[str] = None,
) -> tuple[str, str, str]:
    """
    Returns:
      {
        "thumbprint": <UPPERCASE_SHA1_OF_CERT_DER>,
        "private_key": <PEM PKCS#8 UNENCRYPTED>,
        "public_certificate": <PEM CERT>,
      }
    """
    cred = credential if credential else DefaultAzureCredential()
    secret_client = SecretClient(vault_url=keyvault_url, credential=cred)
    secret = secret_client.get_secret(cert_name)

    # Secret value is often base64-encoded PFX (application/x-pkcs12)
    val = secret.value
    try:
        pfx_bytes = base64.b64decode(val, validate=True)
    except Exception:
        # If not valid base64, treat as raw bytes (rare)
        pfx_bytes = val.encode("latin1")

    password_bytes = pfx_password.encode("utf-8") if pfx_password else None
    private_key, cert, _chain = load_key_and_certificates(pfx_bytes, password_bytes)
    if not private_key or not cert:
        raise ValueError("PFX did not contain both a private key and a certificate.")

    private_key_pem = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    ).decode("utf-8")

    public_cert_pem = cert.public_bytes(Encoding.PEM).decode("utf-8")
    thumbprint_hex = hashlib.sha1(cert.public_bytes(Encoding.DER)).hexdigest().upper()

    return thumbprint_hex, private_key_pem, public_cert_pem

def acquire_token_with_cert(
    tenant_id: str,
    client_id: str,
    private_key_pem: str,
    public_cert_pem: str,
    thumbprint_hex: str,
    scope: str
) -> dict:
    """
    Build MSAL ConfidentialClientApplication and acquire a token for client credentials flow.
    """
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    app = msal.ConfidentialClientApplication(
        client_id=client_id,
        authority=authority,
        client_credential={
            "thumbprint": thumbprint_hex,
            "private_key": private_key_pem,
            "public_certificate": public_cert_pem,
        },
    )
    # Client credentials flow uses a resource ".default" scope
    result = app.acquire_token_for_client(scopes=[scope])
    return result

def acquire_token_from_kv_cert(
    keyvault_url: str,
    cert_name: str,
    tenant_id: str,
    client_id: str,
    scope: str,
    credential: Optional[TokenCredential] = None,
    pfx_password: Optional[str] = None,
) -> dict:
    """
    Retrieve certificate from Azure Key Vault and use it to acquire an access token.

    Args:
        keyvault_url: URL of the Azure Key Vault
        cert_name: Name of the certificate in Key Vault
        tenant_id: Azure AD tenant ID
        client_id: Azure AD application (client) ID
        scope: Scope for the access token (usually a resource URL with ".default")
        credential: Optional TokenCredential for Key Vault access
        pfx_password: Optional password if the PFX is password-protected

    Returns:
        Token response dictionary from MSAL
    """
    thumbprint, private_key_pem, public_cert_pem = build_msal_cert_credentials_from_kv(
        keyvault_url, cert_name, credential, pfx_password
    )
    token_response = acquire_token_with_cert(
        tenant_id, client_id, private_key_pem, public_cert_pem, thumbprint, scope
    )
    return token_response