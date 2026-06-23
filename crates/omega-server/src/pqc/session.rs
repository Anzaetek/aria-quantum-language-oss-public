//! PQC session key derivation and AES-256-GCM encrypted framing.
//!
//! Key derivation: HKDF-SHA256(shared_secret, transcript_hash) → (session_key, session_iv).
//! Frame encryption: AES-256-GCM with nonce = session_iv XOR frame_counter.

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use hkdf::Hkdf;
use sha2::Sha256;

/// Derived session keys for bidirectional encrypted communication.
pub struct SessionKeys {
    cipher: Aes256Gcm,
    iv: [u8; 12],
    send_counter: u64,
    recv_counter: u64,
}

impl SessionKeys {
    /// Derive session keys from the KEM shared secret and handshake transcript.
    ///
    /// `shared_secret`: 32-byte ML-KEM-768 shared secret.
    /// `transcript`: concatenation of all handshake messages (ServerHello || ClientHello).
    pub fn derive(shared_secret: &[u8], transcript: &[u8]) -> Result<Self, String> {
        // HKDF extract + expand
        let hk = Hkdf::<Sha256>::new(None, shared_secret);

        // Derive 32-byte session key
        let mut session_key = [0u8; 32];
        hk.expand(
            &[b"omega-pqc-v1 session key", transcript].concat(),
            &mut session_key,
        )
        .map_err(|e| format!("HKDF expand key: {e}"))?;

        // Derive 12-byte session IV
        let mut iv = [0u8; 12];
        hk.expand(&[b"omega-pqc-v1 session iv", transcript].concat(), &mut iv)
            .map_err(|e| format!("HKDF expand iv: {e}"))?;

        let cipher =
            Aes256Gcm::new_from_slice(&session_key).map_err(|e| format!("AES init: {e}"))?;

        Ok(Self {
            cipher,
            iv,
            send_counter: 0,
            recv_counter: 0,
        })
    }

    /// Compute nonce = session_iv XOR counter (little-endian).
    fn nonce(&self, counter: u64) -> Nonce<aes_gcm::aead::consts::U12> {
        let mut nonce = self.iv;
        let counter_bytes = counter.to_le_bytes();
        for i in 0..8 {
            nonce[i] ^= counter_bytes[i];
        }
        *Nonce::from_slice(&nonce)
    }

    /// Encrypt a plaintext frame. Returns ciphertext + 16-byte auth tag.
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, String> {
        let nonce = self.nonce(self.send_counter);
        self.send_counter += 1;
        self.cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| format!("encrypt: {e}"))
    }

    /// Decrypt a ciphertext frame.
    pub fn decrypt(&mut self, ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        let nonce = self.nonce(self.recv_counter);
        self.recv_counter += 1;
        self.cipher
            .decrypt(&nonce, ciphertext)
            .map_err(|e| format!("decrypt: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let shared_secret = [0x42u8; 32];
        let transcript = b"ServerHello||ClientHello";

        let mut alice = SessionKeys::derive(&shared_secret, transcript).unwrap();
        let mut bob = SessionKeys::derive(&shared_secret, transcript).unwrap();

        let msg = b"quantum circuit execution result";
        let ct = alice.encrypt(msg).unwrap();
        let pt = bob.decrypt(&ct).unwrap();
        assert_eq!(pt, msg);
    }

    #[test]
    fn test_multiple_frames() {
        let shared_secret = [0xAB; 32];
        let transcript = b"tx";

        let mut sender = SessionKeys::derive(&shared_secret, transcript).unwrap();
        let mut receiver = SessionKeys::derive(&shared_secret, transcript).unwrap();

        for i in 0..10 {
            let msg = format!("frame {i}");
            let ct = sender.encrypt(msg.as_bytes()).unwrap();
            let pt = receiver.decrypt(&ct).unwrap();
            assert_eq!(pt, msg.as_bytes());
        }
    }

    #[test]
    fn test_wrong_key_fails() {
        let transcript = b"tx";

        let mut alice = SessionKeys::derive(&[0x11; 32], transcript).unwrap();
        let mut bob = SessionKeys::derive(&[0x22; 32], transcript).unwrap();

        let ct = alice.encrypt(b"secret").unwrap();
        assert!(bob.decrypt(&ct).is_err());
    }

    #[test]
    fn test_replay_fails() {
        let shared_secret = [0xCC; 32];
        let transcript = b"tx";

        let mut sender = SessionKeys::derive(&shared_secret, transcript).unwrap();
        let mut receiver = SessionKeys::derive(&shared_secret, transcript).unwrap();

        let ct = sender.encrypt(b"msg1").unwrap();
        let _ = receiver.decrypt(&ct).unwrap();
        // Replaying the same ciphertext fails because receiver counter advanced
        assert!(receiver.decrypt(&ct).is_err());
    }

    #[test]
    fn different_transcripts_derive_different_keys() {
        // Transcript binding: same shared secret but different
        // transcripts must produce a session that can't decrypt the
        // other side's frames. Guards against a downgrade attack
        // where an MITM substitutes one handshake transcript for
        // another after KEM completes.
        let shared = [0x55u8; 32];
        let mut alice = SessionKeys::derive(&shared, b"transcript-A").unwrap();
        let mut bob = SessionKeys::derive(&shared, b"transcript-B").unwrap();
        let ct = alice.encrypt(b"hi").unwrap();
        assert!(
            bob.decrypt(&ct).is_err(),
            "different transcript must fail decryption"
        );
    }

    #[test]
    fn tampered_ciphertext_fails_decrypt() {
        // AES-GCM auth tag must catch a single-bit flip anywhere in
        // the ciphertext or tag.
        let shared = [0x33u8; 32];
        let mut alice = SessionKeys::derive(&shared, b"tx").unwrap();
        let mut bob = SessionKeys::derive(&shared, b"tx").unwrap();

        let mut ct = alice.encrypt(b"important payload").unwrap();
        // Flip one bit somewhere in the body (not the tag).
        ct[3] ^= 0x01;
        assert!(
            bob.decrypt(&ct).is_err(),
            "tampered body must fail GCM auth"
        );

        // And again, but flip a bit in the trailing 16-byte auth tag.
        let mut ct = alice.encrypt(b"important payload").unwrap();
        let last = ct.len() - 1;
        ct[last] ^= 0x01;
        assert!(bob.decrypt(&ct).is_err(), "tampered tag must fail GCM auth");
    }

    #[test]
    fn out_of_order_frames_fail() {
        // The receiver's counter advances on every successful decrypt;
        // a frame with the wrong sequence number won't authenticate
        // because its nonce changes.
        let shared = [0x77u8; 32];
        let mut sender = SessionKeys::derive(&shared, b"tx").unwrap();
        let mut receiver = SessionKeys::derive(&shared, b"tx").unwrap();

        let _ct0 = sender.encrypt(b"frame 0").unwrap();
        let ct1 = sender.encrypt(b"frame 1").unwrap();
        // Try to decrypt ct1 before ct0 — receiver's counter is at 0
        // but ct1 was encrypted at counter 1.
        assert!(
            receiver.decrypt(&ct1).is_err(),
            "out-of-order receive must fail GCM auth"
        );
    }

    #[test]
    fn empty_plaintext_round_trips() {
        // Edge case: encrypting empty bytes should still produce a
        // valid ciphertext (just the 16-byte tag) that round-trips.
        let shared = [0x99u8; 32];
        let mut alice = SessionKeys::derive(&shared, b"tx").unwrap();
        let mut bob = SessionKeys::derive(&shared, b"tx").unwrap();

        let ct = alice.encrypt(b"").unwrap();
        assert_eq!(ct.len(), 16, "empty plaintext → just the GCM tag");
        let pt = bob.decrypt(&ct).unwrap();
        assert!(pt.is_empty());
    }

    #[test]
    fn ciphertext_carries_tag_overhead() {
        // AES-GCM appends a 16-byte authentication tag. Pin this so a
        // future cipher swap (e.g. XChaCha20-Poly1305) is a deliberate
        // call rather than a silent change in framing overhead.
        let shared = [0xEEu8; 32];
        let mut alice = SessionKeys::derive(&shared, b"tx").unwrap();
        let plaintext = b"hello";
        let ct = alice.encrypt(plaintext).unwrap();
        assert_eq!(ct.len(), plaintext.len() + 16);
    }

    #[test]
    fn nonce_xors_counter_into_iv_lower_8_bytes() {
        // The nonce derivation is `IV XOR LE(counter)` over the
        // first 8 bytes only. Pin this directly so a regression in
        // the byte-order or width is caught here rather than through
        // a flaky end-to-end test.
        let shared = [0xCCu8; 32];
        let session = SessionKeys::derive(&shared, b"tx").unwrap();
        let n0 = session.nonce(0);
        let n1 = session.nonce(1);
        let n_max = session.nonce(u64::MAX);

        // counter=0 → nonce == iv exactly.
        assert_eq!(n0.as_slice(), &session.iv);
        // counter=1 → only byte 0 differs, by 1.
        assert_eq!(n1[0], session.iv[0] ^ 1);
        for i in 1..12 {
            assert_eq!(n1[i], session.iv[i]);
        }
        // counter=u64::MAX → first 8 bytes XOR 0xFF, last 4 unchanged.
        for i in 0..8 {
            assert_eq!(n_max[i], session.iv[i] ^ 0xFF);
        }
        for i in 8..12 {
            assert_eq!(n_max[i], session.iv[i]);
        }
    }

    #[test]
    fn derive_is_deterministic_for_same_inputs() {
        // The HKDF derivation must produce byte-identical session
        // keys for the same (shared_secret, transcript) — both sides
        // of the handshake rely on this matching exactly.
        let shared = [0x44u8; 32];
        let s1 = SessionKeys::derive(&shared, b"deterministic-tx").unwrap();
        let s2 = SessionKeys::derive(&shared, b"deterministic-tx").unwrap();
        assert_eq!(s1.iv, s2.iv);
        // Encrypt/decrypt across the two: must round-trip.
        let mut a = s1;
        let mut b = s2;
        let ct = a.encrypt(b"determinism check").unwrap();
        assert_eq!(b.decrypt(&ct).unwrap(), b"determinism check");
    }
}
