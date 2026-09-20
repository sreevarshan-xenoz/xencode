use serde::{Deserialize, Serialize};

/// Maximum members (distinct identities) in one collaboration session.
/// Shared so the server's enforcement and any client's messaging cannot
/// drift apart.
pub const MAX_SESSION_MEMBERS: usize = 10;

/// WebSocket close codes for handshake failures (application range).
pub const CLOSE_BAD_TOKEN: u16 = 4401;
pub const CLOSE_RBAC_DENIED: u16 = 4403;
pub const CLOSE_NO_SESSION: u16 = 4404;
pub const CLOSE_SESSION_FULL: u16 = 4409;

/// One person in a session, with their workspace role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberInfo {
    pub username: String,
    pub role: String,
}

/// Frames a client may send. Frame #1 must be `Auth`; everything before it
/// is refused without further ceremony.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientFrame {
    Auth { token: String },
    Activity { text: String },
    Ping,
}

/// Frames the server may send. `Activity.user` is always set by the server
/// from its own authenticated state — a client cannot spoof who sent it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerFrame {
    AuthOk {
        username: String,
        role: String,
        session_id: String,
        members: Vec<MemberInfo>,
    },
    Members {
        members: Vec<MemberInfo>,
    },
    Activity {
        user: String,
        text: String,
    },
    Error {
        code: String,
        message: String,
    },
    Pong,
}

impl ClientFrame {
    /// Decode an inbound client frame. Any syntax error or unknown `type`
    /// comes back as `Err` — callers answer with an `Error` frame rather
    /// than panicking on hostile input.
    pub fn parse(raw: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(raw)
    }
}

impl ServerFrame {
    pub fn to_json(&self) -> String {
        // Every variant serializes; a failure here would be a bug in the
        // type, not in user input.
        serde_json::to_string(self).expect("ServerFrame always serializes")
    }

    pub fn error(code: &str, message: &str) -> Self {
        ServerFrame::Error {
            code: code.to_string(),
            message: message.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(frame: &ClientFrame) -> ClientFrame {
        let json = serde_json::to_string(frame).unwrap();
        ClientFrame::parse(&json).unwrap()
    }

    #[test]
    fn client_frames_round_trip() {
        let frames = vec![
            ClientFrame::Auth {
                token: "xencode_abc".to_string(),
            },
            ClientFrame::Activity {
                text: "fn main() {}".to_string(),
            },
            ClientFrame::Ping,
        ];
        for frame in frames {
            assert_eq!(round_trip(&frame), frame);
        }
    }

    #[test]
    fn server_frames_round_trip_through_json() {
        let frame = ServerFrame::AuthOk {
            username: "alice".to_string(),
            role: "editor".to_string(),
            session_id: "xencode-1234".to_string(),
            members: vec![
                MemberInfo {
                    username: "alice".to_string(),
                    role: "editor".to_string(),
                },
                MemberInfo {
                    username: "bob".to_string(),
                    role: "admin".to_string(),
                },
            ],
        };
        let parsed: ServerFrame = serde_json::from_str(&frame.to_json()).unwrap();
        assert_eq!(parsed, frame);

        for frame in [
            ServerFrame::Members {
                members: vec![MemberInfo {
                    username: "alice".to_string(),
                    role: "viewer".to_string(),
                }],
            },
            ServerFrame::Activity {
                user: "alice".to_string(),
                text: "hi".to_string(),
            },
            ServerFrame::error("rbac_denied", "viewers cannot relay"),
            ServerFrame::Pong,
        ] {
            let parsed: ServerFrame = serde_json::from_str(&frame.to_json()).unwrap();
            assert_eq!(parsed, frame);
        }
    }

    /// The protocol's hostile-input contract: garbage yields Err, never a
    /// panic, and never a silently-wrong frame.
    #[test]
    fn unknown_or_malformed_frames_are_errors_not_panics() {
        assert!(ClientFrame::parse(r#"{"type":"bogus"}"#).is_err());
        assert!(ClientFrame::parse(r#"{"no_type":1}"#).is_err());
        assert!(ClientFrame::parse("not json").is_err());
        assert!(ClientFrame::parse("").is_err());
        // Right type, wrong shape: the tag must not paper over bad payloads.
        assert!(ClientFrame::parse(r#"{"type":"auth"}"#).is_err());
    }

    #[test]
    fn frames_carry_the_type_tag_on_the_wire() {
        let json = serde_json::to_string(&ClientFrame::Ping).unwrap();
        assert_eq!(json, r#"{"type":"ping"}"#);
        let json = serde_json::to_string(&ServerFrame::Pong).unwrap();
        assert_eq!(json, r#"{"type":"pong"}"#);
    }

    #[test]
    fn close_codes_are_in_the_application_range() {
        for code in [
            CLOSE_BAD_TOKEN,
            CLOSE_RBAC_DENIED,
            CLOSE_NO_SESSION,
            CLOSE_SESSION_FULL,
        ] {
            assert!((4000..5000).contains(&code));
        }
    }
}
