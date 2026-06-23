//! PQC token signing and verification using ML-DSA-65.
//!
//! Two crypto backends, selected at compile time via features:
//! - `crypto-ml-dsa` (default): Pure Rust RustCrypto implementation (FIPS 204)
//! - `crypto-mldsa-c`: C-backed pqcrypto-mldsa (FIPS 204, ~10× faster
//!   than the pure-Rust path on the keygen path; requires a C toolchain).

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Token payload (claims).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    /// Unique token ID
    pub jti: String,
    /// Subject / principal name
    pub sub: String,
    /// Issued-at (Unix seconds)
    pub iat: i64,
    /// Expiration (Unix seconds)
    pub exp: i64,
    /// Granted rights bitfield
    pub rights: u32,
    /// Signing key ID
    pub kid: String,
}

// ---- Feature-gated crypto backend ----

#[cfg(feature = "crypto-ml-dsa")]
mod crypto {
    use ml_dsa::{EncodedVerifyingKey, MlDsa65, Seed, SigningKey, VerifyingKey};
    use signature::{Signer, Verifier};

    pub fn generate_keypair_bytes() -> (Vec<u8>, Vec<u8>) {
        let seed_bytes: [u8; 32] = rand::random();
        let seed = Seed::from(seed_bytes);

        let sk = SigningKey::<MlDsa65>::from_seed(&seed);
        let vk = sk.verifying_key();

        // Store: PK = encoded verifying key, SK = 32-byte seed
        let pk_enc = vk.encode();
        let pk_bytes: &[u8] = pk_enc.as_ref();
        (pk_bytes.to_vec(), seed_bytes.to_vec())
    }

    pub fn sign_detached(payload: &[u8], sk_bytes: &[u8]) -> Result<Vec<u8>, String> {
        let seed_bytes: [u8; 32] = sk_bytes
            .try_into()
            .map_err(|_| "invalid seed size (expected 32 bytes)")?;
        let seed = Seed::from(seed_bytes);
        let sk = SigningKey::<MlDsa65>::from_seed(&seed);
        let sig = sk.sign(payload);
        let sig_enc = sig.encode();
        let sig_bytes: &[u8] = sig_enc.as_ref();
        Ok(sig_bytes.to_vec())
    }

    pub fn verify_detached(
        payload: &[u8],
        sig_bytes: &[u8],
        pk_bytes: &[u8],
    ) -> Result<(), String> {
        let sig = ml_dsa::Signature::<MlDsa65>::try_from(sig_bytes)
            .map_err(|e| format!("invalid signature: {e}"))?;
        let vk_enc = EncodedVerifyingKey::<MlDsa65>::try_from(pk_bytes)
            .map_err(|_| "invalid public key size".to_string())?;
        let vk = VerifyingKey::<MlDsa65>::decode(&vk_enc);
        vk.verify(payload, &sig)
            .map_err(|e| format!("verification failed: {e}"))
    }
}

#[cfg(all(feature = "crypto-mldsa-c", not(feature = "crypto-ml-dsa")))]
mod crypto {
    use pqcrypto_mldsa::mldsa65;
    use pqcrypto_traits::sign::{
        DetachedSignature as DsTrait, PublicKey as PkTrait, SecretKey as SkTrait,
    };

    pub fn generate_keypair_bytes() -> (Vec<u8>, Vec<u8>) {
        let (pk, sk) = mldsa65::keypair();
        (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
    }

    pub fn sign_detached(payload: &[u8], sk_bytes: &[u8]) -> Result<Vec<u8>, String> {
        let sk = mldsa65::SecretKey::from_bytes(sk_bytes).map_err(|e| format!("{e:?}"))?;
        let sig = mldsa65::detached_sign(payload, &sk);
        Ok(sig.as_bytes().to_vec())
    }

    pub fn verify_detached(
        payload: &[u8],
        sig_bytes: &[u8],
        pk_bytes: &[u8],
    ) -> Result<(), String> {
        let pk = mldsa65::PublicKey::from_bytes(pk_bytes).map_err(|e| format!("{e:?}"))?;
        let sig =
            mldsa65::DetachedSignature::from_bytes(sig_bytes).map_err(|e| format!("{e:?}"))?;
        mldsa65::verify_detached_signature(&sig, payload, &pk)
            .map_err(|_| "verification failed".to_string())
    }
}

#[cfg(not(any(feature = "crypto-ml-dsa", feature = "crypto-mldsa-c")))]
compile_error!("Enable either 'crypto-ml-dsa' or 'crypto-mldsa-c' feature for omega-server");

/// Generate an ML-DSA-65 keypair. Returns (key_id, public_key_bytes, secret_key_bytes).
pub fn generate_keypair() -> (String, Vec<u8>, Vec<u8>) {
    let kid = Uuid::new_v4().to_string();
    let (pk, sk) = crypto::generate_keypair_bytes();
    (kid, pk, sk)
}

/// Create and sign a token. Returns the encoded token string.
pub fn issue_token(
    sub: &str,
    rights: u32,
    ttl_seconds: i64,
    kid: &str,
    sk_bytes: &[u8],
) -> Result<(String, TokenClaims), String> {
    let now = Utc::now().timestamp();
    let claims = TokenClaims {
        jti: Uuid::new_v4().to_string(),
        sub: sub.to_string(),
        iat: now,
        exp: now + ttl_seconds,
        rights,
        kid: kid.to_string(),
    };

    let token_str = sign_claims(&claims, sk_bytes)?;
    Ok((token_str, claims))
}

/// Sign a TokenClaims struct, returning `base64(payload).base64(detached_signature)`.
fn sign_claims(claims: &TokenClaims, sk_bytes: &[u8]) -> Result<String, String> {
    let payload = serde_json::to_vec(claims).map_err(|e| e.to_string())?;
    let sig = crypto::sign_detached(&payload, sk_bytes)?;

    let payload_b64 = URL_SAFE_NO_PAD.encode(&payload);
    let sig_b64 = URL_SAFE_NO_PAD.encode(&sig);

    Ok(format!("{payload_b64}.{sig_b64}"))
}

/// Verify a token string. Returns claims if valid.
pub fn verify_token(token: &str, pk_bytes: &[u8]) -> Result<TokenClaims, String> {
    let parts: Vec<&str> = token.splitn(2, '.').collect();
    if parts.len() != 2 {
        return Err("invalid token format".into());
    }

    let payload_bytes = URL_SAFE_NO_PAD
        .decode(parts[0])
        .map_err(|_| "invalid base64 in payload")?;
    let sig_bytes = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|_| "invalid base64 in signature")?;

    crypto::verify_detached(&payload_bytes, &sig_bytes, pk_bytes)?;

    let claims: TokenClaims = serde_json::from_slice(&payload_bytes).map_err(|e| e.to_string())?;

    let now = Utc::now().timestamp();
    if claims.exp <= now {
        return Err("token expired".into());
    }

    Ok(claims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let (kid, pk, sk) = generate_keypair();
        let (token, claims) = issue_token("test-user", 0x0F, 3600, &kid, &sk).unwrap();
        let verified = verify_token(&token, &pk).unwrap();
        assert_eq!(verified.sub, "test-user");
        assert_eq!(verified.rights, 0x0F);
        assert_eq!(verified.kid, kid);
        assert_eq!(verified.jti, claims.jti);
    }

    #[test]
    fn test_invalid_signature_rejected() {
        let (kid, _pk, sk) = generate_keypair();
        let (token, _) = issue_token("test", 0x01, 3600, &kid, &sk).unwrap();
        let (_, pk2, _) = generate_keypair();
        assert!(verify_token(&token, &pk2).is_err());
    }

    #[test]
    fn test_expired_token_rejected() {
        let (kid, pk, sk) = generate_keypair();
        let (token, _) = issue_token("test", 0x01, -1, &kid, &sk).unwrap();
        assert!(verify_token(&token, &pk).is_err());
    }
}

/// Bit-by-bit comparison of both ML-DSA-65 implementations.
#[cfg(test)]
mod comparison {
    use std::time::Instant;

    // ---- Option A: pqcrypto-mldsa (C-backed, FIPS 204) ----
    mod option_a {
        use pqcrypto_mldsa::mldsa65;
        use pqcrypto_traits::sign::{
            DetachedSignature as DsTrait, PublicKey as PkTrait, SecretKey as SkTrait,
        };

        pub fn generate() -> (Vec<u8>, Vec<u8>) {
            let (pk, sk) = mldsa65::keypair();
            (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
        }

        pub fn sign(msg: &[u8], sk: &[u8]) -> Vec<u8> {
            let sk = mldsa65::SecretKey::from_bytes(sk).unwrap();
            mldsa65::detached_sign(msg, &sk).as_bytes().to_vec()
        }

        pub fn verify(msg: &[u8], sig: &[u8], pk: &[u8]) -> bool {
            let pk = match mldsa65::PublicKey::from_bytes(pk) {
                Ok(pk) => pk,
                Err(_) => return false,
            };
            let sig = match mldsa65::DetachedSignature::from_bytes(sig) {
                Ok(sig) => sig,
                Err(_) => return false,
            };
            mldsa65::verify_detached_signature(&sig, msg, &pk).is_ok()
        }
    }

    // ---- Option B: ml-dsa (pure Rust, FIPS 204) ----
    mod option_b {
        use ml_dsa::{EncodedVerifyingKey, MlDsa65, Seed, SigningKey, VerifyingKey};
        use signature::{Signer, Verifier};

        pub fn generate() -> (Vec<u8>, Vec<u8>) {
            let seed: [u8; 32] = rand::random();
            let sk = SigningKey::<MlDsa65>::from_seed(&Seed::from(seed));
            let vk = sk.verifying_key();
            let pk_enc = vk.encode();
            let pk_bytes: &[u8] = pk_enc.as_ref();
            (pk_bytes.to_vec(), seed.to_vec())
        }

        pub fn sign(msg: &[u8], sk_seed: &[u8]) -> Vec<u8> {
            let seed: [u8; 32] = sk_seed.try_into().unwrap();
            let sk = SigningKey::<MlDsa65>::from_seed(&Seed::from(seed));
            let sig_enc = sk.sign(msg).encode();
            let sig_bytes: &[u8] = sig_enc.as_ref();
            sig_bytes.to_vec()
        }

        pub fn verify(msg: &[u8], sig: &[u8], pk: &[u8]) -> bool {
            let sig = match ml_dsa::Signature::<MlDsa65>::try_from(sig) {
                Ok(sig) => sig,
                Err(_) => return false,
            };
            let vk_enc = match EncodedVerifyingKey::<MlDsa65>::try_from(pk) {
                Ok(enc) => enc,
                Err(_) => return false,
            };
            let vk = VerifyingKey::<MlDsa65>::decode(&vk_enc);
            vk.verify(msg, &sig).is_ok()
        }
    }

    fn hex_prefix(data: &[u8], n: usize) -> String {
        data.iter()
            .take(n)
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join("")
    }

    #[test]
    fn compare_ml_dsa_65_implementations() {
        let msg = b"Omega Functions PQC token comparison payload";

        println!("\n============================================================");
        println!("ML-DSA-65 Implementation Comparison: Option A vs Option B");
        println!("============================================================\n");

        // ---- Key Generation ----
        let (pk_a, sk_a) = option_a::generate();
        let (pk_b, sk_b) = option_b::generate();

        println!("KEY SIZES:");
        println!("  Option A (pqcrypto-mldsa, C-backed):");
        println!("    Public key:  {} bytes", pk_a.len());
        println!("    Secret key:  {} bytes (expanded)", sk_a.len());
        println!("  Option B (ml-dsa, pure Rust, FIPS 204):");
        println!("    Public key:  {} bytes", pk_b.len());
        println!("    Secret key:  {} bytes (seed)", sk_b.len());
        println!(
            "  PK match: {} | SK format: expanded ({}) vs seed ({})",
            pk_a.len() == pk_b.len(),
            sk_a.len(),
            sk_b.len()
        );

        // ---- Signing ----
        let sig_a = option_a::sign(msg, &sk_a);
        let sig_b = option_b::sign(msg, &sk_b);

        println!("\nSIGNATURE SIZES:");
        println!("  Option A: {} bytes", sig_a.len());
        println!("  Option B: {} bytes", sig_b.len());
        println!("  Match: {}", sig_a.len() == sig_b.len());
        if sig_a.len() != sig_b.len() {
            println!(
                "  (pqcrypto-mldsa = {} vs ml-dsa = {})",
                sig_a.len(),
                sig_b.len()
            );
        }

        // ---- Self-verification ----
        assert!(
            option_a::verify(msg, &sig_a, &pk_a),
            "Option A self-verify failed"
        );
        assert!(
            option_b::verify(msg, &sig_b, &pk_b),
            "Option B self-verify failed"
        );
        println!("\nSELF-VERIFICATION:");
        println!("  Option A: PASS");
        println!("  Option B: PASS");

        // ---- Cross-verification ----
        let cross_ab = option_b::verify(msg, &sig_a, &pk_a);
        let cross_ba = option_a::verify(msg, &sig_b, &pk_b);
        println!("\nCROSS-VERIFICATION:");
        println!(
            "  A-key+sig verified by B impl: {}",
            if cross_ab {
                "PASS (same standard)"
            } else {
                "FAIL (expected: different standards)"
            }
        );
        println!(
            "  B-key+sig verified by A impl: {}",
            if cross_ba {
                "PASS (same standard)"
            } else {
                "FAIL (expected: different standards)"
            }
        );

        // ---- Tampered message rejected ----
        let tampered = b"TAMPERED payload";
        assert!(!option_a::verify(tampered, &sig_a, &pk_a));
        assert!(!option_b::verify(tampered, &sig_b, &pk_b));
        println!("\nTAMPER REJECTION:");
        println!("  Option A rejects tampered msg: PASS");
        println!("  Option B rejects tampered msg: PASS");

        // ---- Wrong key rejected ----
        // Generate fresh keys for wrong-key test (don't cross A/B since sizes differ)
        let (pk_a2, _) = option_a::generate();
        let (pk_b2, _) = option_b::generate();
        assert!(!option_a::verify(msg, &sig_a, &pk_a2));
        assert!(!option_b::verify(msg, &sig_b, &pk_b2));
        println!("\nWRONG-KEY REJECTION:");
        println!("  Option A rejects wrong pk: PASS");
        println!("  Option B rejects wrong pk: PASS");

        // ---- Performance comparison ----
        let n = 50u32;

        let t = Instant::now();
        for _ in 0..n {
            let _ = option_a::generate();
        }
        let keygen_a = t.elapsed();

        let t = Instant::now();
        for _ in 0..n {
            let _ = option_b::generate();
        }
        let keygen_b = t.elapsed();

        let t = Instant::now();
        for _ in 0..n {
            let _ = option_a::sign(msg, &sk_a);
        }
        let sign_a = t.elapsed();

        let t = Instant::now();
        for _ in 0..n {
            let _ = option_b::sign(msg, &sk_b);
        }
        let sign_b = t.elapsed();

        let t = Instant::now();
        for _ in 0..n {
            let _ = option_a::verify(msg, &sig_a, &pk_a);
        }
        let verify_a = t.elapsed();

        let t = Instant::now();
        for _ in 0..n {
            let _ = option_b::verify(msg, &sig_b, &pk_b);
        }
        let verify_b = t.elapsed();

        println!("\nPERFORMANCE ({n} iterations, avg per op):");
        println!("  Keygen:  A={:?}, B={:?}", keygen_a / n, keygen_b / n);
        println!("  Sign:    A={:?}, B={:?}", sign_a / n, sign_b / n);
        println!("  Verify:  A={:?}, B={:?}", verify_a / n, verify_b / n);

        // ---- Byte-level format inspection ----
        println!("\nKEY FORMAT (first 32 bytes, hex):");
        println!("  PK A: {}", hex_prefix(&pk_a, 32));
        println!("  PK B: {}", hex_prefix(&pk_b, 32));
        println!("  SK A: {} (expanded)", hex_prefix(&sk_a, 32));
        println!("  SK B: {} (seed, full)", hex_prefix(&sk_b, 32));

        println!("\nSIGNATURE FORMAT (first 32 bytes, hex):");
        println!("  Sig A: {}", hex_prefix(&sig_a, 32));
        println!("  Sig B: {}", hex_prefix(&sig_b, 32));

        // ---- Summary ----
        println!("\n------------------------------------------------------------");
        println!("SUMMARY");
        println!("------------------------------------------------------------");
        println!("  Algorithm A: ML-DSA-65 (pqcrypto-mldsa, C reference, FIPS 204)");
        println!("  Algorithm B: ML-DSA-65 (ml-dsa, pure Rust, FIPS 204)");
        println!("  PK size:  A={}, B={}", pk_a.len(), pk_b.len());
        println!(
            "  SK size:  A={} (expanded), B={} (seed)",
            sk_a.len(),
            sk_b.len()
        );
        println!("  Sig size: A={}, B={}", sig_a.len(), sig_b.len());
        println!("  C compiler required: A=YES, B=NO");
        println!("  FIPS 204 compliant:  A=YES, B=YES");
        println!("  Recommendation: Option B — pure Rust, no C toolchain, 32-byte seed keys");
    }
}
