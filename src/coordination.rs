//! Multi-core coordination profiles for synchronization barriers (US-02 R1, ADR-006, ADR-007).

use core::fmt;
use core::str::FromStr;

use crate::error::K3Error;

/// Multi-core worker coordination profile for layer and operator synchronization barriers.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum CoordinationProfile {
    /// Atomic busy-polling barrier with `core::hint::spin_loop()` (lowest latency, 100% CPU utilization).
    #[default]
    Spin,
    /// Kernel-assisted thread sleep yielding CPU cycles via futex / Condvar.
    Futex,
    /// Dynamic task scheduling via a dedicated scoped Rayon work-stealing thread pool.
    Rayon,
}

impl CoordinationProfile {
    /// Parses a coordination profile string ("spin", "futex", "rayon").
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if the profile string is unrecognized.
    pub fn parse(name: &str) -> Result<Self, K3Error> {
        name.parse()
    }

    /// Returns the canonical CLI string identifier for this profile.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Spin => "spin",
            Self::Futex => "futex",
            Self::Rayon => "rayon",
        }
    }

    /// Returns an array of all available coordination profiles.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::Spin, Self::Futex, Self::Rayon]
    }
}

impl fmt::Display for CoordinationProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for CoordinationProfile {
    type Err = K3Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "spin" => Ok(Self::Spin),
            "futex" => Ok(Self::Futex),
            "rayon" => Ok(Self::Rayon),
            other => Err(K3Error::CapabilityMismatch {
                architecture: "coordination".to_owned(),
                reason: format!(
                    "unknown coordination profile '{other}': expected 'spin', 'futex', or 'rayon'"
                ),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordination_profile_parsing_and_display() {
        assert_eq!(CoordinationProfile::default(), CoordinationProfile::Spin);
        assert_eq!(CoordinationProfile::Spin.as_str(), "spin");
        assert_eq!(CoordinationProfile::Futex.as_str(), "futex");
        assert_eq!(CoordinationProfile::Rayon.as_str(), "rayon");

        assert_eq!(CoordinationProfile::Spin.to_string(), "spin");
        assert_eq!(CoordinationProfile::Futex.to_string(), "futex");
        assert_eq!(CoordinationProfile::Rayon.to_string(), "rayon");

        assert_eq!(
            "spin".parse::<CoordinationProfile>().unwrap(),
            CoordinationProfile::Spin
        );
        assert_eq!(
            "SPIN".parse::<CoordinationProfile>().unwrap(),
            CoordinationProfile::Spin
        );
        assert_eq!(
            " futex ".parse::<CoordinationProfile>().unwrap(),
            CoordinationProfile::Futex
        );
        assert_eq!(
            "Rayon".parse::<CoordinationProfile>().unwrap(),
            CoordinationProfile::Rayon
        );

        let err = "invalid".parse::<CoordinationProfile>().unwrap_err();
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
        assert_eq!(err.diagnostic_code(), "capability_mismatch");

        let all = CoordinationProfile::all();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], CoordinationProfile::Spin);
        assert_eq!(all[1], CoordinationProfile::Futex);
        assert_eq!(all[2], CoordinationProfile::Rayon);
    }

    #[test]
    fn test_coordination_profile_parse_method() {
        assert_eq!(
            CoordinationProfile::parse("spin").unwrap(),
            CoordinationProfile::Spin
        );
        assert_eq!(
            CoordinationProfile::parse("FUTEX").unwrap(),
            CoordinationProfile::Futex
        );
        assert_eq!(
            CoordinationProfile::parse("rayon").unwrap(),
            CoordinationProfile::Rayon
        );
        assert!(CoordinationProfile::parse("unknown").is_err());
    }
}
