//! Where the badge sits: the place it was last dragged to, if that is still on
//! the screen, otherwise the middle of the right edge. Saved as
//! `<settings dir>/badge.json`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The badge's size in logical pixels.
pub const SIZE: f32 = 40.0;
/// Gap kept between the badge and the screen edge.
pub const MARGIN: f32 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Saved {
    pub x: f32,
    pub y: f32,
}

/// A screen rectangle: origin and size, in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Screen {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub fn settings_file() -> Option<PathBuf> {
    xencode_config_rs::paths::settings_dir()
        .ok()
        .map(|dir| dir.join("badge.json"))
}

pub fn load(file: &Path) -> Option<Saved> {
    serde_json::from_str(&std::fs::read_to_string(file).ok()?).ok()
}

pub fn save(file: &Path, at: Saved) -> std::io::Result<()> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(
        file,
        serde_json::to_string(&at).map_err(std::io::Error::other)?,
    )
}

/// Where to open: the saved place when the whole badge is still on `screen`
/// (a monitor may have been unplugged since), else the right edge, halfway down.
pub fn initial(saved: Option<Saved>, screen: Screen) -> Saved {
    if let Some(at) = saved {
        let fits = at.x >= screen.x
            && at.y >= screen.y
            && at.x + SIZE <= screen.x + screen.width
            && at.y + SIZE <= screen.y + screen.height;
        if fits {
            return at;
        }
    }
    Saved {
        x: screen.x + screen.width - SIZE - MARGIN,
        y: screen.y + (screen.height - SIZE) / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: Screen = Screen {
        x: 0.0,
        y: 0.0,
        width: 2560.0,
        height: 1080.0,
    };

    #[test]
    fn with_nothing_saved_it_opens_halfway_down_the_right_edge() {
        assert_eq!(
            initial(None, SCREEN),
            Saved {
                x: 2560.0 - SIZE - MARGIN,
                y: (1080.0 - SIZE) / 2.0
            }
        );
    }

    #[test]
    fn a_saved_place_on_the_screen_is_kept() {
        let at = Saved { x: 100.0, y: 200.0 };
        assert_eq!(initial(Some(at), SCREEN), at);
    }

    #[test]
    fn a_saved_place_off_the_screen_falls_back_to_the_edge() {
        let gone = Saved {
            x: 3000.0,
            y: 200.0,
        };
        assert_eq!(initial(Some(gone), SCREEN), initial(None, SCREEN));
        let half_off = Saved {
            x: 2540.0,
            y: 200.0,
        };
        assert_eq!(initial(Some(half_off), SCREEN), initial(None, SCREEN));
    }

    #[test]
    fn a_saved_place_round_trips_and_a_broken_file_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("nested").join("badge.json");
        save(&file, Saved { x: 12.5, y: 40.0 }).unwrap();
        assert_eq!(load(&file), Some(Saved { x: 12.5, y: 40.0 }));
        std::fs::write(&file, "{").unwrap();
        assert_eq!(load(&file), None);
    }
}
