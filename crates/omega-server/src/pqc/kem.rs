//! ML-KEM-768 key encapsulation wrapper.

use ml_kem::kem::{Decapsulate, Encapsulate, Kem, KeyExport, TryKeyInit};
use ml_kem::{DecapsulationKey768, EncapsulationKey768, MlKem768};

/// KEM keypair: decapsulation key (secret) + encapsulation key (public).
pub struct KemKeypair {
    pub dk: DecapsulationKey768,
    pub ek: EncapsulationKey768,
}

impl KemKeypair {
    /// Generate a fresh ML-KEM-768 keypair.
    pub fn generate() -> Self {
        let (dk, ek) = MlKem768::generate_keypair();
        Self { dk, ek }
    }

    /// Get the encapsulation key bytes (public, 1184 bytes).
    pub fn ek_bytes(&self) -> Vec<u8> {
        self.ek.to_bytes().to_vec()
    }
}

/// Encapsulate a shared secret to a remote encapsulation key.
/// Returns (ciphertext_bytes, shared_secret_32_bytes).
#[allow(dead_code)]
pub fn encapsulate(ek_bytes: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ek = EncapsulationKey768::new_from_slice(ek_bytes)
        .map_err(|_| "invalid encapsulation key size".to_string())?;
    let (ct, ss) = ek.encapsulate();
    let ct_bytes: &[u8] = ct.as_ref();
    let ss_bytes: &[u8] = ss.as_ref();
    Ok((ct_bytes.to_vec(), ss_bytes.to_vec()))
}

/// Decapsulate a shared secret from a ciphertext.
pub fn decapsulate(dk: &DecapsulationKey768, ct_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let ss = dk
        .decapsulate_slice(ct_bytes)
        .map_err(|e| format!("decapsulation failed: {e}"))?;
    let ss_bytes: &[u8] = ss.as_ref();
    Ok(ss_bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kem_roundtrip() {
        let kp = KemKeypair::generate();
        let ek_bytes = kp.ek_bytes();

        let (ct_bytes, ss_encap) = encapsulate(&ek_bytes).unwrap();
        let ss_decap = decapsulate(&kp.dk, &ct_bytes).unwrap();

        assert_eq!(ss_encap.len(), 32);
        assert_eq!(ss_decap.len(), 32);
        assert_eq!(ss_encap, ss_decap, "shared secrets must match");
    }

    #[test]
    fn test_wrong_dk_produces_different_ss() {
        let kp1 = KemKeypair::generate();
        let kp2 = KemKeypair::generate();

        let (ct_bytes, ss1) = encapsulate(&kp1.ek_bytes()).unwrap();
        let ss2 = decapsulate(&kp2.dk, &ct_bytes).unwrap();

        // ML-KEM implicit rejection: wrong DK returns pseudorandom value, not error
        assert_ne!(ss1, ss2, "wrong DK should produce different shared secret");
    }

    #[test]
    fn ek_bytes_has_ml_kem_768_size() {
        // FIPS 203 ML-KEM-768: encapsulation key is 1184 bytes.
        // Pin the size so a crate swap (or accidental ML-KEM-512/1024
        // selection) is caught here rather than as a wire mismatch
        // against existing clients.
        let kp = KemKeypair::generate();
        assert_eq!(kp.ek_bytes().len(), 1184);
    }

    #[test]
    fn shared_secret_is_32_bytes() {
        // FIPS 203: the KEM shared secret is 32 bytes regardless of
        // the parameter set. Both encap and decap must yield this
        // exact length.
        let kp = KemKeypair::generate();
        let (ct, ss_e) = encapsulate(&kp.ek_bytes()).unwrap();
        let ss_d = decapsulate(&kp.dk, &ct).unwrap();
        assert_eq!(ss_e.len(), 32);
        assert_eq!(ss_d.len(), 32);
    }

    #[test]
    fn ciphertext_has_ml_kem_768_size() {
        // FIPS 203 ML-KEM-768: ciphertext is 1088 bytes.
        let kp = KemKeypair::generate();
        let (ct, _ss) = encapsulate(&kp.ek_bytes()).unwrap();
        assert_eq!(ct.len(), 1088);
    }

    #[test]
    fn encapsulate_rejects_wrong_size_ek() {
        // A truncated or oversized public key must surface as an
        // error rather than producing nonsense (or worse, panicking
        // in the underlying crate).
        let err = encapsulate(b"too-short").expect_err("undersized ek");
        assert!(err.contains("encapsulation key"), "msg: {err}");

        // Oversized: ML-KEM-768 EK is 1184 bytes; pad to 2000.
        let mut oversized = vec![0u8; 2000];
        let kp = KemKeypair::generate();
        oversized[..1184].copy_from_slice(&kp.ek_bytes());
        let err = encapsulate(&oversized).expect_err("oversized ek");
        assert!(err.contains("encapsulation key"), "msg: {err}");
    }

    #[test]
    fn decapsulate_rejects_wrong_size_ciphertext() {
        // A truncated CT can either be rejected with an error or, in
        // some implementations, returned as a valid-but-junk SS via
        // implicit rejection. The pure-Rust ml-kem crate raises an
        // error for size mismatch.
        let kp = KemKeypair::generate();
        let result = decapsulate(&kp.dk, b"too-short");
        assert!(result.is_err(), "undersized CT must error");
        let err = result.unwrap_err();
        assert!(
            err.contains("decapsulation"),
            "expected decapsulation error, got {err}"
        );
    }

    #[test]
    fn fresh_keypairs_are_distinct() {
        // Two consecutive keypair generations must produce different
        // public keys with overwhelming probability — guards against a
        // catastrophic regression where the entropy source is wired
        // wrong (e.g. seeded from a constant).
        let kp1 = KemKeypair::generate();
        let kp2 = KemKeypair::generate();
        assert_ne!(kp1.ek_bytes(), kp2.ek_bytes());
    }

    #[test]
    fn ek_bytes_round_trip_via_encapsulate() {
        // Pull `ek_bytes()` out, hand them to `encapsulate`, and
        // verify the resulting CT decapsulates correctly with the
        // original DK. This is the actual path the WS handshake
        // uses: server bytes EK → client encapsulates → server
        // decapsulates with stored DK.
        let kp = KemKeypair::generate();
        let ek_bytes = kp.ek_bytes();
        let (ct, ss_client) = encapsulate(&ek_bytes).unwrap();
        let ss_server = decapsulate(&kp.dk, &ct).unwrap();
        assert_eq!(ss_client, ss_server, "round-trip via byte form must match");
    }
}
