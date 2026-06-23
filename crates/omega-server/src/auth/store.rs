//! SQLite persistence for signing keys and token metadata.

use rusqlite::{params, Connection};

use super::token::TokenClaims;

/// Create auth tables if they don't exist.
pub fn init_auth_tables(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS signing_keys (
            id TEXT PRIMARY KEY,
            public_key BLOB NOT NULL,
            secret_key BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            revoked_at TEXT
        );
        CREATE TABLE IF NOT EXISTS tokens (
            jti TEXT PRIMARY KEY,
            sub TEXT NOT NULL,
            kid TEXT NOT NULL REFERENCES signing_keys(id),
            rights INTEGER NOT NULL,
            issued_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            revoked_at TEXT
        );",
    )
    .map_err(|e| e.to_string())
}

/// Get the active (non-revoked) signing key, or generate one if none exists.
/// Returns (kid, public_key_bytes, secret_key_bytes).
pub fn ensure_signing_key(conn: &Connection) -> Result<(String, Vec<u8>, Vec<u8>), String> {
    let mut stmt = conn
        .prepare("SELECT id, public_key, secret_key FROM signing_keys WHERE revoked_at IS NULL ORDER BY created_at DESC LIMIT 1")
        .map_err(|e| e.to_string())?;

    let result = stmt.query_row([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Vec<u8>>(1)?,
            row.get::<_, Vec<u8>>(2)?,
        ))
    });

    match result {
        Ok(key) => Ok(key),
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            let (kid, pk, sk) = super::token::generate_keypair();
            conn.execute(
                "INSERT INTO signing_keys (id, public_key, secret_key) VALUES (?1, ?2, ?3)",
                params![kid, pk, sk],
            )
            .map_err(|e| e.to_string())?;
            Ok((kid, pk, sk))
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Store a token record for revocation tracking.
pub fn store_token(conn: &Connection, claims: &TokenClaims) -> Result<(), String> {
    conn.execute(
        "INSERT INTO tokens (jti, sub, kid, rights, issued_at, expires_at) VALUES (?1, ?2, ?3, ?4, datetime(?5, 'unixepoch'), datetime(?6, 'unixepoch'))",
        params![claims.jti, claims.sub, claims.kid, claims.rights, claims.iat, claims.exp],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Revoke a token by JTI.
pub fn revoke_token(conn: &Connection, jti: &str) -> Result<bool, String> {
    let affected = conn
        .execute(
            "UPDATE tokens SET revoked_at = datetime('now') WHERE jti = ?1 AND revoked_at IS NULL",
            params![jti],
        )
        .map_err(|e| e.to_string())?;
    Ok(affected > 0)
}

/// Check if a token is revoked.
pub fn is_token_revoked(conn: &Connection, jti: &str) -> Result<bool, String> {
    let result = conn.query_row(
        "SELECT revoked_at FROM tokens WHERE jti = ?1",
        params![jti],
        |row| row.get::<_, Option<String>>(0),
    );

    match result {
        Ok(Some(_)) => Ok(true),                                // has revoked_at
        Ok(None) => Ok(false),                                  // not revoked
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(false), // not tracked = not revoked
        Err(e) => Err(e.to_string()),
    }
}

/// Get the public key for a given key ID.
pub fn get_public_key(conn: &Connection, kid: &str) -> Result<Vec<u8>, String> {
    conn.query_row(
        "SELECT public_key FROM signing_keys WHERE id = ?1 AND revoked_at IS NULL",
        params![kid],
        |row| row.get(0),
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::rights;
    use crate::auth::token::issue_token;

    fn fresh_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        init_auth_tables(&conn).expect("init tables");
        conn
    }

    #[test]
    fn init_auth_tables_is_idempotent() {
        // Calling init twice on the same connection must not fail —
        // matches the boot path's behaviour when the registry is
        // re-opened against an existing on-disk DB.
        let conn = Connection::open_in_memory().unwrap();
        init_auth_tables(&conn).expect("first init");
        init_auth_tables(&conn).expect("second init");
    }

    #[test]
    fn ensure_signing_key_creates_then_returns_same_key() {
        let conn = fresh_db();
        let (kid1, pk1, sk1) = ensure_signing_key(&conn).expect("first call inserts");
        let (kid2, pk2, sk2) = ensure_signing_key(&conn).expect("second call reads back");
        assert_eq!(kid1, kid2);
        assert_eq!(pk1, pk2);
        assert_eq!(sk1, sk2);
    }

    #[test]
    fn store_and_revoke_token_round_trip() {
        let conn = fresh_db();
        let (kid, _pk, sk) = ensure_signing_key(&conn).unwrap();
        let (_token_str, claims) =
            issue_token("alice", rights::READ, 60, &kid, &sk).expect("issue");

        store_token(&conn, &claims).expect("store");
        // Not revoked yet.
        assert!(!is_token_revoked(&conn, &claims.jti).unwrap());

        // First revoke flips the flag.
        assert!(revoke_token(&conn, &claims.jti).expect("revoke"));
        assert!(is_token_revoked(&conn, &claims.jti).unwrap());

        // Re-revoking the same JTI is a no-op (already-revoked rows
        // are filtered out by the UPDATE clause).
        assert!(!revoke_token(&conn, &claims.jti).unwrap());
    }

    #[test]
    fn is_token_revoked_unknown_jti_returns_false() {
        // "not tracked" must be treated as "not revoked" — the auth
        // middleware is responsible for rejecting unknown JTIs via
        // signature verification, not via this query.
        let conn = fresh_db();
        assert!(!is_token_revoked(&conn, "no-such-jti").unwrap());
    }

    #[test]
    fn get_public_key_returns_active_key() {
        let conn = fresh_db();
        let (kid, pk, _sk) = ensure_signing_key(&conn).unwrap();
        let fetched = get_public_key(&conn, &kid).expect("fetch active key");
        assert_eq!(fetched, pk);
    }

    #[test]
    fn get_public_key_unknown_kid_errors() {
        let conn = fresh_db();
        let err = get_public_key(&conn, "no-such-kid").expect_err("unknown kid");
        assert!(!err.is_empty(), "expected non-empty error message");
    }

    #[test]
    fn store_token_rejects_duplicate_jti() {
        // PRIMARY KEY on jti must reject re-insertion of the same JTI —
        // a guarantee the issue path relies on (UUID collisions are
        // not assumed; the DB enforces uniqueness).
        let conn = fresh_db();
        let (kid, _pk, sk) = ensure_signing_key(&conn).unwrap();
        let (_t, claims) = issue_token("alice", rights::READ, 60, &kid, &sk).expect("issue");
        store_token(&conn, &claims).expect("first store");
        let err = store_token(&conn, &claims).expect_err("duplicate JTI must reject");
        assert!(
            err.contains("UNIQUE") || err.contains("constraint"),
            "expected UNIQUE constraint error, got {err}"
        );
    }
}
