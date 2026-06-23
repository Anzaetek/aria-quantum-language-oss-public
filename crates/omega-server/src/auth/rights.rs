//! Bitfield-based rights model for API authorization.

pub const READ: u32 = 1 << 0;
pub const EXECUTE: u32 = 1 << 1;
pub const WRITE: u32 = 1 << 2;
pub const ADMIN: u32 = 1 << 3;

// Predefined roles
#[allow(dead_code)]
pub const VIEWER: u32 = READ;
#[allow(dead_code)]
pub const OPERATOR: u32 = READ | EXECUTE;
#[allow(dead_code)]
pub const DEVELOPER: u32 = READ | EXECUTE | WRITE;
pub const ADMIN_ROLE: u32 = READ | EXECUTE | WRITE | ADMIN;

#[inline]
pub fn has_right(granted: u32, required: u32) -> bool {
    (granted & required) == required
}

pub fn rights_to_names(r: u32) -> Vec<&'static str> {
    let mut names = Vec::new();
    if r & READ != 0 {
        names.push("read");
    }
    if r & EXECUTE != 0 {
        names.push("execute");
    }
    if r & WRITE != 0 {
        names.push("write");
    }
    if r & ADMIN != 0 {
        names.push("admin");
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_constants_compose_their_bits() {
        assert_eq!(VIEWER, READ);
        assert_eq!(OPERATOR, READ | EXECUTE);
        assert_eq!(DEVELOPER, READ | EXECUTE | WRITE);
        assert_eq!(ADMIN_ROLE, READ | EXECUTE | WRITE | ADMIN);
    }

    #[test]
    fn has_right_requires_every_bit() {
        // Subset checks: VIEWER has READ, doesn't have EXECUTE.
        assert!(has_right(VIEWER, READ));
        assert!(!has_right(VIEWER, EXECUTE));
        assert!(!has_right(VIEWER, READ | EXECUTE));

        // ADMIN_ROLE clears every flag including ADMIN.
        assert!(has_right(ADMIN_ROLE, READ));
        assert!(has_right(ADMIN_ROLE, EXECUTE));
        assert!(has_right(ADMIN_ROLE, WRITE));
        assert!(has_right(ADMIN_ROLE, ADMIN));
        assert!(has_right(ADMIN_ROLE, READ | WRITE | EXECUTE | ADMIN));
    }

    #[test]
    fn has_right_zero_required_is_always_true() {
        // Required = 0 means "no rights required" — should always pass.
        assert!(has_right(0, 0));
        assert!(has_right(VIEWER, 0));
        assert!(has_right(ADMIN_ROLE, 0));
    }

    #[test]
    fn has_right_with_no_grants_is_always_false_when_required_nonzero() {
        assert!(!has_right(0, READ));
        assert!(!has_right(0, ADMIN));
    }

    #[test]
    fn rights_to_names_preserves_bit_order() {
        assert_eq!(rights_to_names(0), Vec::<&str>::new());
        assert_eq!(rights_to_names(READ), vec!["read"]);
        assert_eq!(rights_to_names(EXECUTE), vec!["execute"]);
        assert_eq!(rights_to_names(WRITE), vec!["write"]);
        assert_eq!(rights_to_names(ADMIN), vec!["admin"]);
        // Compound: ordering is bit-position, not insertion order.
        assert_eq!(
            rights_to_names(ADMIN_ROLE),
            vec!["read", "execute", "write", "admin"]
        );
        assert_eq!(rights_to_names(OPERATOR), vec!["read", "execute"]);
    }
}
